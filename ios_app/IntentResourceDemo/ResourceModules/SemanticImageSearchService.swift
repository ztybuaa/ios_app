import CoreImage
import CoreML
import Foundation
import Photos
import UIKit

final class SemanticImageSearchService {
    static let shared = SemanticImageSearchService()

    private let ciContext = CIContext()
    private let rgbColorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    private let models: Result<ChineseCLIPModels, Error>
    private let cacheLock = NSLock()
    private let predictionLock = NSLock()
    private var imageEmbeddingCache: [String: [NamedEmbedding]] = [:]
    private var textEmbeddingCache: [String: [Float]] = [:]

    private init() {
        do {
            models = .success(try ChineseCLIPModels(bundle: .main))
        } catch {
            models = .failure(error)
        }
    }

    func search(
        assets: [PHAsset],
        kind: CandidateKind,
        slots: NormalizedSlots,
        limit: Int
    ) async throws -> [ResourceCandidate] {
        let models = try loadedModels()
        let query = SemanticQueryPlanner.query(for: slots)
        let positiveEmbeddings = try textEmbeddings(for: query.positivePrompts, models: models)
        let negativeEmbeddings = try textEmbeddings(for: query.negativePrompts, models: models)

        var candidates: [ResourceCandidate] = []
        let scanAssets = Array(assets.prefix(semanticScanLimit(for: slots)))
        let batchSize = 2

        for batchStart in stride(from: 0, to: scanAssets.count, by: batchSize) {
            if Task.isCancelled { break }
            let batchEnd = min(batchStart + batchSize, scanAssets.count)
            let batch = Array(scanAssets[batchStart..<batchEnd])

            try await withThrowingTaskGroup(of: ResourceCandidate?.self) { group in
                for asset in batch {
                    group.addTask {
                        try await self.matchCandidate(
                            asset: asset,
                            kind: kind,
                            query: query,
                            positiveEmbeddings: positiveEmbeddings,
                            negativeEmbeddings: negativeEmbeddings,
                            models: models
                        )
                    }
                }

                for try await candidate in group {
                    if let candidate {
                        candidates.append(candidate)
                    }
                }
            }
        }

        return candidates
            .sorted { $0.score == $1.score ? $0.title < $1.title : $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }

    private func loadedModels() throws -> ChineseCLIPModels {
        switch models {
        case .success(let models):
            return models
        case .failure(let error):
            throw DemoError.resourceUnavailable(error.localizedDescription)
        }
    }

    private func semanticScanLimit(for slots: NormalizedSlots) -> Int {
        if slots.qualifiers.selectionHint.contains("all") {
            return 3_000
        }
        if slots.resourcePhrase.contains("最近") || slots.qualifiers.selectionHint.contains("recent") {
            return 600
        }
        return 2_000
    }

    private func matchCandidate(
        asset: PHAsset,
        kind: CandidateKind,
        query: SemanticQuery,
        positiveEmbeddings: [[Float]],
        negativeEmbeddings: [[Float]],
        models: ChineseCLIPModels
    ) async throws -> ResourceCandidate? {
        guard let imageEmbeddings = try await imageEmbeddings(for: asset, query: query, models: models),
              !imageEmbeddings.isEmpty else {
            return nil
        }

        var bestMatch = SemanticMatchScore(
            positive: -1,
            negative: -1,
            margin: -.infinity,
            cropName: "none"
        )
        for imageEmbedding in imageEmbeddings {
            let positive = positiveEmbeddings
                .map { cosineSimilarity(imageEmbedding.values, $0) }
                .max() ?? -1
            let negative = negativeEmbeddings
                .map { cosineSimilarity(imageEmbedding.values, $0) }
                .max() ?? -1
            let margin = positive - negative
            if margin > bestMatch.margin || (margin == bestMatch.margin && positive > bestMatch.positive) {
                bestMatch = SemanticMatchScore(
                    positive: positive,
                    negative: negative,
                    margin: margin,
                    cropName: imageEmbedding.name
                )
            }
        }

        guard bestMatch.positive >= query.minimumSimilarity,
              bestMatch.margin >= query.minimumMargin else {
            return nil
        }

        let score = Double(bestMatch.margin * 1_000 + bestMatch.positive * 10)
        return makeCandidate(
            asset: asset,
            kind: kind,
            score: score,
            match: bestMatch,
            prompts: query.positivePrompts
        )
    }

    private func textEmbeddings(
        for prompts: [String],
        models: ChineseCLIPModels
    ) throws -> [[Float]] {
        guard !prompts.isEmpty else {
            throw DemoError.resourceUnavailable("Chinese-CLIP 没有收到有效中文提示词。")
        }

        var embeddings: [[Float]] = []
        embeddings.reserveCapacity(prompts.count)
        for prompt in prompts {
            if let cached = cachedTextEmbedding(prompt) {
                embeddings.append(cached)
                continue
            }
            let embedding = try computeTextEmbedding(prompt, models: models)
            setCachedTextEmbedding(embedding, for: prompt)
            embeddings.append(embedding)
        }
        return embeddings
    }

    private func computeTextEmbedding(
        _ prompt: String,
        models: ChineseCLIPModels
    ) throws -> [Float] {
        let tokenIDs = models.tokenizer.encode(prompt)
        let input = try MLMultiArray(
            shape: [1, 52],
            dataType: .int32
        )
        for (index, tokenID) in tokenIDs.enumerated() {
            input[index] = NSNumber(value: tokenID)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: ["text": input])
        let output = try predict(models.text, provider: provider)
        guard let features = output.featureValue(for: "text_features")?.multiArrayValue else {
            throw ChineseCLIPError.missingPredictionOutput("text_features")
        }
        return try normalizedEmbedding(features.floatValues, source: "text_features")
    }

    private func imageEmbeddings(
        for asset: PHAsset,
        query: SemanticQuery,
        models: ChineseCLIPModels
    ) async throws -> [NamedEmbedding]? {
        let cacheKey = "\(asset.localIdentifier)|\(query.embeddingProfile)"
        if let cached = cachedImageEmbeddings(cacheKey) {
            return cached
        }
        guard let image = await requestImage(asset: asset, needsRegionScan: query.needsRegionScan),
              let imageTensors = try imageTensors(from: image, needsRegionScan: query.needsRegionScan) else {
            return nil
        }

        var embeddings: [NamedEmbedding] = []
        embeddings.reserveCapacity(imageTensors.count)
        for item in imageTensors {
            let provider = try MLDictionaryFeatureProvider(dictionary: ["image": item.tensor])
            let output = try predict(models.image, provider: provider)
            guard let features = output.featureValue(for: "image_features")?.multiArrayValue else {
                throw ChineseCLIPError.missingPredictionOutput("image_features")
            }
            let values = try normalizedEmbedding(features.floatValues, source: "image_features")
            embeddings.append(NamedEmbedding(name: item.name, values: values))
        }

        guard !embeddings.isEmpty else {
            return nil
        }
        setCachedImageEmbeddings(embeddings, for: cacheKey)
        return embeddings
    }

    private func predict(
        _ model: MLModel,
        provider: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        predictionLock.lock()
        defer { predictionLock.unlock() }
        return try model.prediction(from: provider)
    }

    private func requestImage(asset: PHAsset, needsRegionScan: Bool) async -> UIImage? {
        await withCheckedContinuation { continuation in
            let lock = NSLock()
            var didResume = false

            func resumeOnce(_ image: UIImage?) {
                lock.lock()
                defer { lock.unlock() }
                guard !didResume else { return }
                didResume = true
                continuation.resume(returning: image)
            }

            let options = PHImageRequestOptions()
            options.deliveryMode = .highQualityFormat
            options.resizeMode = .exact
            options.isNetworkAccessAllowed = false

            let side = needsRegionScan ? 900 : 512
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: CGSize(width: side, height: side),
                contentMode: .aspectFit,
                options: options
            ) { image, info in
                if let cancelled = info?[PHImageCancelledKey] as? Bool, cancelled {
                    resumeOnce(nil)
                    return
                }
                if info?[PHImageErrorKey] != nil {
                    resumeOnce(nil)
                    return
                }
                resumeOnce(image)
            }
        }
    }

    private func imageTensors(
        from image: UIImage,
        needsRegionScan: Bool
    ) throws -> [NamedImageTensor]? {
        guard let sourceImage = CIImage(image: image)?.translatedToOrigin() else {
            return nil
        }

        let targetSize = CGSize(width: ChineseCLIPModelContract.imageSize, height: ChineseCLIPModelContract.imageSize)
        var views: [(String, CIImage)] = []
        if let full = sourceImage.resizeBicubic(size: targetSize) {
            views.append(("full", full))
        }
        if let center = sourceImage.cropToSquare(at: .center)?.resizeBicubic(size: targetSize) {
            views.append(("center", center))
        }
        if needsRegionScan {
            for anchor in RegionAnchor.cornerAnchors {
                if let crop = sourceImage.cropToSquare(at: anchor)?.resizeBicubic(size: targetSize) {
                    views.append((anchor.rawValue, crop))
                }
            }
        }

        var result: [NamedImageTensor] = []
        var seen = Set<String>()
        for view in views where seen.insert(view.0).inserted {
            result.append(NamedImageTensor(name: view.0, tensor: try imageTensor(from: view.1)))
        }
        return result
    }

    private func imageTensor(from image: CIImage) throws -> MLMultiArray {
        let size = ChineseCLIPModelContract.imageSize
        let rowBytes = size * 4
        var pixels = Array(repeating: UInt8(0), count: rowBytes * size)
        pixels.withUnsafeMutableBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            ciContext.render(
                image,
                toBitmap: baseAddress,
                rowBytes: rowBytes,
                bounds: CGRect(x: 0, y: 0, width: size, height: size),
                format: .RGBA8,
                colorSpace: rgbColorSpace
            )
        }

        let tensor = try MLMultiArray(
            shape: [1, 3, 224, 224],
            dataType: .float32
        )
        let tensorValues = tensor.dataPointer.bindMemory(to: Float.self, capacity: tensor.count)
        let planeSize = size * size
        pixels.withUnsafeBytes { bytes in
            let rgba = bytes.bindMemory(to: UInt8.self)
            for pixelIndex in 0..<planeSize {
                let byteIndex = pixelIndex * 4
                let red = Float(rgba[byteIndex]) / 255
                let green = Float(rgba[byteIndex + 1]) / 255
                let blue = Float(rgba[byteIndex + 2]) / 255
                tensorValues[pixelIndex] = (red - 0.48145466) / 0.26862954
                tensorValues[planeSize + pixelIndex] = (green - 0.4578275) / 0.26130258
                tensorValues[planeSize * 2 + pixelIndex] = (blue - 0.40821073) / 0.27577711
            }
        }
        return tensor
    }

    private func makeCandidate(
        asset: PHAsset,
        kind: CandidateKind,
        score: Double,
        match: SemanticMatchScore,
        prompts: [String]
    ) -> ResourceCandidate {
        let date = asset.creationDate.map {
            DateFormatter.localizedString(from: $0, dateStyle: .short, timeStyle: .short)
        } ?? "未知时间"
        let dimensions = "\(asset.pixelWidth)x\(asset.pixelHeight)"
        let title = kind == .video ? "视频 \(date)" : "照片 \(date)"
        let promptPreview = prompts.prefix(4).joined(separator: " / ")
        let detail = String(
            format: "语义差值 %.3f，正向 %.3f，负向 %.3f，视图 %@，提示词：%@",
            Double(match.margin),
            Double(match.positive),
            Double(match.negative),
            match.cropName,
            promptPreview
        )

        return ResourceCandidate(
            id: asset.localIdentifier,
            kind: kind,
            title: title,
            subtitle: dimensions,
            detail: detail,
            score: score,
            debugInfo: "Chinese-CLIP RN50 FP16 中文语义检索"
        )
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else {
            return -1
        }
        return zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
    }

    private func normalizedEmbedding(_ values: [Float], source: String) throws -> [Float] {
        guard values.count == ChineseCLIPModelContract.embeddingDimensions,
              values.allSatisfy(\.isFinite) else {
            throw ChineseCLIPError.invalidEmbedding(source, values.count)
        }
        let norm = sqrt(values.reduce(Float(0)) { $0 + $1 * $1 })
        guard norm.isFinite, norm > 0 else {
            throw ChineseCLIPError.invalidEmbedding(source, values.count)
        }
        return values.map { $0 / norm }
    }

    private func cachedImageEmbeddings(_ key: String) -> [NamedEmbedding]? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return imageEmbeddingCache[key]
    }

    private func setCachedImageEmbeddings(_ value: [NamedEmbedding], for key: String) {
        cacheLock.lock()
        imageEmbeddingCache[key] = value
        cacheLock.unlock()
    }

    private func cachedTextEmbedding(_ key: String) -> [Float]? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return textEmbeddingCache[key]
    }

    private func setCachedTextEmbedding(_ value: [Float], for key: String) {
        cacheLock.lock()
        textEmbeddingCache[key] = value
        cacheLock.unlock()
    }
}

private struct ChineseCLIPModels {
    let tokenizer: ChineseCLIPTokenizer
    let image: MLModel
    let text: MLModel

    init(bundle: Bundle) throws {
        tokenizer = try ChineseCLIPTokenizer(bundle: bundle)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        image = try Self.loadModel(
            named: ChineseCLIPModelContract.imageModelName,
            bundle: bundle,
            configuration: configuration
        )
        text = try Self.loadModel(
            named: ChineseCLIPModelContract.textModelName,
            bundle: bundle,
            configuration: configuration
        )

        try Self.validate(
            image,
            input: "image",
            inputShape: [1, 3, 224, 224],
            inputType: .float32,
            output: "image_features"
        )
        try Self.validate(
            text,
            input: "text",
            inputShape: [1, 52],
            inputType: .int32,
            output: "text_features"
        )
    }

    private static func loadModel(
        named name: String,
        bundle: Bundle,
        configuration: MLModelConfiguration
    ) throws -> MLModel {
        guard let url = bundle.url(forResource: name, withExtension: "mlmodelc") else {
            throw ChineseCLIPError.missingModel(name)
        }
        do {
            return try MLModel(contentsOf: url, configuration: configuration)
        } catch {
            throw ChineseCLIPError.unreadableModel(name, error.localizedDescription)
        }
    }

    private static func validate(
        _ model: MLModel,
        input: String,
        inputShape: [Int],
        inputType: MLMultiArrayDataType,
        output: String
    ) throws {
        guard let inputDescription = model.modelDescription.inputDescriptionsByName[input],
              inputDescription.type == .multiArray,
              let inputConstraint = inputDescription.multiArrayConstraint,
              inputConstraint.shape.map({ $0.intValue }) == inputShape,
              inputConstraint.dataType == inputType else {
            throw ChineseCLIPError.invalidModelContract("input \(input) must be \(inputType) \(inputShape)")
        }
        guard let outputDescription = model.modelDescription.outputDescriptionsByName[output],
              outputDescription.type == .multiArray,
              let outputConstraint = outputDescription.multiArrayConstraint,
              outputConstraint.shape.map({ $0.intValue }).reduce(1, *) == ChineseCLIPModelContract.embeddingDimensions else {
            throw ChineseCLIPError.invalidModelContract(
                "output \(output) must contain \(ChineseCLIPModelContract.embeddingDimensions) values"
            )
        }
    }
}

private enum ChineseCLIPModelContract {
    static let textModelName = "chinese_clip_rn50_text"
    static let imageModelName = "chinese_clip_rn50_image"
    static let imageSize = 224
    static let embeddingDimensions = 1_024
    static let profile = "chinese-clip-rn50-fp16-v1"
}

private enum ChineseCLIPError: LocalizedError {
    case missingModel(String)
    case unreadableModel(String, String)
    case invalidModelContract(String)
    case missingPredictionOutput(String)
    case invalidEmbedding(String, Int)

    var errorDescription: String? {
        switch self {
        case .missingModel(let name):
            return "Chinese-CLIP 模型 \(name).mlmodelc 不在 App Bundle 中。"
        case .unreadableModel(let name, let reason):
            return "Chinese-CLIP 模型 \(name) 加载失败：\(reason)"
        case .invalidModelContract(let reason):
            return "Chinese-CLIP RN50 模型接口不匹配：\(reason)。"
        case .missingPredictionOutput(let name):
            return "Chinese-CLIP 推理缺少输出 \(name)。"
        case .invalidEmbedding(let source, let count):
            return "Chinese-CLIP 输出 \(source) 不是有效的 1024 维向量，实际长度为 \(count)。"
        }
    }
}

private enum ChineseCLIPRN50Config {
    static let minimumSimilarity: Float = 0.47
    static let screenshotMinimumMargin: Float = 0.011
    static let defaultMinimumMargin: Float = 0.012
}

private enum SemanticQueryPlanner {
    static func query(for slots: NormalizedSlots) -> SemanticQuery {
        let keyword = slots.searchKeyword?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let phrase = slots.resourcePhrase.trimmingCharacters(in: .whitespacesAndNewlines)
        let subject = normalizedWhitespace(phrase.isEmpty ? keyword : phrase)
        let searchText = "\(subject) \(phrase)".lowercased()

        let wantsScreen = containsAny(searchText, terms: ["截图", "截屏", "屏幕", "界面", "screen", "screenshot"])
        let wantsGame = containsAny(searchText, terms: ["游戏", "手游", "游戏画面", "game", "gameplay"])
        let wantsScenery = containsAny(
            searchText,
            terms: ["风景", "景色", "自然", "户外", "山", "湖", "海滩", "天空", "landscape"]
        )
        let wantsCat = searchText.contains("猫")
        let wantsDog = searchText.contains("狗") || searchText.contains("犬")
        let wantsPerson = containsAny(
            searchText,
            terms: ["美女", "人物", "人像", "女人", "女生", "男人", "男生", "自拍", "合影", "portrait"]
        )

        let negativePrompts: [String]
        if wantsGame {
            negativePrompts = [
                "普通应用界面",
                "网页截图",
                "手机设置界面",
                "小猫",
                "小狗",
                "风景",
                "人像"
            ]
        } else if wantsScreen {
            negativePrompts = ["小猫", "小狗", "风景", "人像", "宠物"]
        } else if wantsCat {
            negativePrompts = ["狗的照片", "野生动物照片", "老虎照片", "风景照片", "人物照片", "手机截图"]
        } else if wantsDog {
            negativePrompts = ["猫的照片", "野生动物照片", "老虎照片", "风景照片", "人物照片", "手机截图"]
        } else if wantsScenery {
            negativePrompts = ["人物照片", "宠物照片", "手机截图", "文档图片"]
        } else if wantsPerson {
            negativePrompts = ["宠物照片", "风景照片", "手机截图", "没有人物的物品照片"]
        } else {
            negativePrompts = ["与查询内容无关的图片", "随机图片", "模糊图片"]
        }

        return SemanticQuery(
            positivePrompts: orderedUnique([subject]),
            negativePrompts: orderedUnique(negativePrompts),
            needsRegionScan: true,
            minimumSimilarity: ChineseCLIPRN50Config.minimumSimilarity,
            minimumMargin: wantsScreen && !wantsGame
                ? ChineseCLIPRN50Config.screenshotMinimumMargin
                : ChineseCLIPRN50Config.defaultMinimumMargin
        )
    }

    private static func normalizedWhitespace(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func orderedUnique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for value in values {
            let normalized = normalizedWhitespace(value)
            guard !normalized.isEmpty else { continue }
            if seen.insert(normalized.lowercased()).inserted {
                result.append(normalized)
            }
        }
        return result
    }

    private static func containsAny(_ text: String, terms: [String]) -> Bool {
        terms.contains { text.contains($0) }
    }
}

private struct SemanticQuery {
    let positivePrompts: [String]
    let negativePrompts: [String]
    let needsRegionScan: Bool
    let minimumSimilarity: Float
    let minimumMargin: Float

    var embeddingProfile: String {
        "\(ChineseCLIPModelContract.profile)-\(needsRegionScan ? "regions" : "full")"
    }
}

private struct NamedEmbedding {
    let name: String
    let values: [Float]
}

private struct NamedImageTensor {
    let name: String
    let tensor: MLMultiArray
}

private struct SemanticMatchScore {
    let positive: Float
    let negative: Float
    let margin: Float
    let cropName: String
}

private enum RegionAnchor: String {
    case center
    case topLeft = "top-left"
    case topRight = "top-right"
    case bottomLeft = "bottom-left"
    case bottomRight = "bottom-right"

    static let cornerAnchors: [RegionAnchor] = [.topLeft, .topRight, .bottomLeft, .bottomRight]
}

private extension CIImage {
    func translatedToOrigin() -> CIImage {
        transformed(
            by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y)
        )
    }

    func cropToSquare(at anchor: RegionAnchor) -> CIImage? {
        guard extent.width > 0, extent.height > 0 else {
            return nil
        }
        let size = min(extent.width, extent.height)
        let x: CGFloat
        let y: CGFloat

        switch anchor {
        case .center:
            x = round((extent.width - size) / 2)
            y = round((extent.height - size) / 2)
        case .topLeft:
            x = 0
            y = max(0, extent.height - size)
        case .topRight:
            x = max(0, extent.width - size)
            y = max(0, extent.height - size)
        case .bottomLeft:
            x = 0
            y = 0
        case .bottomRight:
            x = max(0, extent.width - size)
            y = 0
        }

        let rect = CGRect(x: x, y: y, width: size, height: size)
        return cropped(to: rect)
            .transformed(by: CGAffineTransform(translationX: -x, y: -y))
    }

    func resizeBicubic(size targetSize: CGSize) -> CIImage? {
        guard extent.width > 0, extent.height > 0,
              targetSize.width > 0, targetSize.height > 0,
              let filter = CIFilter(name: "CIBicubicScaleTransform") else {
            return nil
        }
        let source = translatedToOrigin()
        let scale = targetSize.height / source.extent.height
        let aspectRatio = (targetSize.width / source.extent.width) / scale
        filter.setValue(source, forKey: kCIInputImageKey)
        filter.setValue(scale, forKey: kCIInputScaleKey)
        filter.setValue(aspectRatio, forKey: kCIInputAspectRatioKey)
        guard let output = filter.outputImage else {
            return nil
        }
        return output.cropped(to: CGRect(origin: .zero, size: targetSize))
    }
}

private extension MLMultiArray {
    var floatValues: [Float] {
        var values: [Float] = []
        values.reserveCapacity(count)
        for index in 0..<count {
            values.append(self[index].floatValue)
        }
        return values
    }
}
