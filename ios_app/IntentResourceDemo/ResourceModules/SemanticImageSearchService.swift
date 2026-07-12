import CryptoKit
import CoreImage
import CoreML
import Foundation
import Photos
import UIKit

enum SemanticSearchMode {
    case photo
    case videoPoster(subject: String)
}

final class SemanticImageSearchService {
    static let shared = SemanticImageSearchService()

    private let ciContext = CIContext()
    private let rgbColorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
    private let models: Result<ChineseCLIPModels, Error>
    private let embeddingStore = ImageEmbeddingStore(namespace: ChineseCLIPModelContract.profile)
    private let cacheLock = NSLock()
    private let predictionLock = NSLock()
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
        limit: Int,
        mode: SemanticSearchMode = .photo
    ) async throws -> SemanticSearchOutcome {
        let searchStarted = CFAbsoluteTimeGetCurrent()
        let models = try loadedModels()
        let query: SemanticQuery
        switch mode {
        case .photo:
            query = SemanticQueryPlanner.query(for: slots)
        case .videoPoster(let subject):
            query = SemanticQueryPlanner.videoPosterQuery(subject: subject)
        }
        let positiveEmbeddings = try textEmbeddings(for: query.positivePrompts, models: models)
        let negativeEmbeddings = try textEmbeddings(for: query.negativePrompts, models: models)
        let scanAssets = Array(assets.prefix(semanticScanLimit(for: slots)))
        let metrics = SemanticSearchMetricsCollector()

        let fullPassStarted = CFAbsoluteTimeGetCurrent()
        let coarseMatches = try await matches(
            assets: scanAssets,
            profile: .coarse,
            positiveEmbeddings: positiveEmbeddings,
            negativeEmbeddings: negativeEmbeddings,
            models: models,
            metrics: metrics
        )
        let fullPassWallMs = elapsedMs(since: fullPassStarted)

        let rerankStarted = CFAbsoluteTimeGetCurrent()
        let qualityShortlist = Array(
            coarseMatches
                .sorted(by: isRankedBefore)
                .prefix(semanticQualityRerankLimit(resultLimit: limit))
        )
        let qualityMatches = try await matches(
            assets: qualityShortlist.map(\.asset),
            profile: .full,
            positiveEmbeddings: positiveEmbeddings,
            negativeEmbeddings: negativeEmbeddings,
            models: models,
            metrics: metrics
        )

        let rerankWallMs = elapsedMs(since: rerankStarted)

        let candidates = qualityMatches
            .filter {
                $0.match.positive >= query.minimumSimilarity &&
                $0.match.margin >= query.minimumMargin
            }
            .sorted(by: isRankedBefore)
            .prefix(limit)
            .map {
                makeCandidate(
                    asset: $0.asset,
                    kind: kind,
                    score: score(for: $0.match),
                    match: $0.match,
                    prompts: query.positivePrompts
                )
            }

        return SemanticSearchOutcome(
            candidates: candidates,
            metrics: metrics.snapshot(
                modelLoadTimeMs: models.loadTimeMs,
                totalWallMs: elapsedMs(since: searchStarted),
                fullPassWallMs: fullPassWallMs,
                rerankWallMs: rerankWallMs,
                scannedAssetCount: scanAssets.count,
                shortlistAssetCount: qualityShortlist.count
            )
        )
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

    private func semanticQualityRerankLimit(resultLimit: Int) -> Int {
        min(128, max(64, resultLimit * 4))
    }

    private func matches(
        assets: [PHAsset],
        profile: ImageEmbeddingProfile,
        positiveEmbeddings: [[Float]],
        negativeEmbeddings: [[Float]],
        models: ChineseCLIPModels,
        metrics: SemanticSearchMetricsCollector
    ) async throws -> [SemanticAssetMatch] {
        var matches: [SemanticAssetMatch] = []
        let batchSize = 2

        for batchStart in stride(from: 0, to: assets.count, by: batchSize) {
            if Task.isCancelled { break }
            let batchEnd = min(batchStart + batchSize, assets.count)
            let batch = Array(assets[batchStart..<batchEnd])

            try await withThrowingTaskGroup(of: SemanticAssetMatch?.self) { group in
                for asset in batch {
                    group.addTask {
                        guard let embeddings = try await self.imageEmbeddings(
                            for: asset,
                            profile: profile,
                            models: models,
                            metrics: metrics
                        ) else {
                            return nil
                        }
                        return SemanticAssetMatch(
                            asset: asset,
                            match: self.bestMatch(
                                embeddings: embeddings,
                                positiveEmbeddings: positiveEmbeddings,
                                negativeEmbeddings: negativeEmbeddings
                            )
                        )
                    }
                }

                for try await match in group {
                    if let match {
                        matches.append(match)
                    }
                }
            }
        }
        return matches
    }

    private func bestMatch(
        embeddings: [NamedEmbedding],
        positiveEmbeddings: [[Float]],
        negativeEmbeddings: [[Float]]
    ) -> SemanticMatchScore {
        var bestMatch = SemanticMatchScore(
            positive: -1,
            negative: -1,
            margin: -.infinity,
            cropName: "none"
        )
        for imageEmbedding in embeddings {
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
        return bestMatch
    }

    private func isRankedBefore(_ lhs: SemanticAssetMatch, _ rhs: SemanticAssetMatch) -> Bool {
        if lhs.match.margin != rhs.match.margin {
            return lhs.match.margin > rhs.match.margin
        }
        if lhs.match.positive != rhs.match.positive {
            return lhs.match.positive > rhs.match.positive
        }
        return lhs.asset.localIdentifier < rhs.asset.localIdentifier
    }

    private func score(for match: SemanticMatchScore) -> Double {
        Double(match.margin * 1_000 + match.positive * 10)
    }

    private func elapsedMs(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1_000
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
        profile: ImageEmbeddingProfile,
        models: ChineseCLIPModels,
        metrics: SemanticSearchMetricsCollector
    ) async throws -> [NamedEmbedding]? {
        let cacheKey = ImageEmbeddingCacheKey(asset: asset, profile: profile)
        switch embeddingStore.lookup(cacheKey) {
        case .memory(let cached):
            metrics.recordMemoryHit()
            return cached
        case .disk(let cached):
            metrics.recordDiskHit()
            return cached
        case .corrupt:
            metrics.recordCorruptEviction()
        case .unavailable:
            metrics.recordStorageUnavailable()
        case .miss:
            break
        }
        metrics.recordCacheMiss()

        guard let image = await requestImage(asset: asset, profile: profile),
              let imageTensors = try imageTensors(from: image, profile: profile) else {
            metrics.recordImageRequestFailure()
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
        metrics.recordPredictions(profile: profile, count: embeddings.count)

        guard !embeddings.isEmpty else {
            return nil
        }
        embeddingStore.store(embeddings, for: cacheKey)
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

    private func requestImage(asset: PHAsset, profile: ImageEmbeddingProfile) async -> UIImage? {
        let manager = PHImageManager.default()
        let state = PhotoImageRequestState()
        return await withTaskCancellationHandler(operation: {
            await withCheckedContinuation { continuation in
                guard state.install(continuation) else { return }
                let options = PHImageRequestOptions()
                switch profile {
                case .coarse:
                    options.deliveryMode = .fastFormat
                    options.resizeMode = .fast
                case .full:
                    options.deliveryMode = .highQualityFormat
                    options.resizeMode = .exact
                }
                options.version = .current
                options.isNetworkAccessAllowed = false

                let requestID = manager.requestImage(
                    for: asset,
                    targetSize: CGSize(width: profile.requestSide, height: profile.requestSide),
                    contentMode: .aspectFit,
                    options: options
                ) { image, info in
                    if let cancelled = info?[PHImageCancelledKey] as? Bool, cancelled {
                        state.complete(nil)
                        return
                    }
                    if info?[PHImageErrorKey] != nil {
                        state.complete(nil)
                        return
                    }
                    if let degraded = info?[PHImageResultIsDegradedKey] as? Bool, degraded {
                        if profile == .coarse {
                            state.complete(image)
                        }
                        return
                    }
                    state.complete(image)
                }
                state.bind(requestID: requestID, manager: manager)
                DispatchQueue.global(qos: .utility).asyncAfter(
                    deadline: .now() + profile.requestTimeout
                ) {
                    state.cancel()
                }
            }
        }, onCancel: {
            state.cancel()
        })
    }

    private func imageTensors(
        from image: UIImage,
        profile: ImageEmbeddingProfile
    ) throws -> [NamedImageTensor]? {
        guard let sourceImage = CIImage(image: image)?.translatedToOrigin() else {
            return nil
        }

        let targetSize = CGSize(width: ChineseCLIPModelContract.imageSize, height: ChineseCLIPModelContract.imageSize)
        var views: [(String, CIImage)] = []
        switch profile {
        case .coarse, .full:
            if let full = sourceImage.resizeBicubic(size: targetSize) {
                views.append(("full", full))
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
    let loadTimeMs: Double

    init(bundle: Bundle) throws {
        let started = CFAbsoluteTimeGetCurrent()
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
        loadTimeMs = (CFAbsoluteTimeGetCurrent() - started) * 1_000
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
    static let profile = "chinese-clip-rn50-b196ee3e-fp16-preprocess-v2-two-stage-quality-v1"
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

        return query(subject: subject, searchText: searchText)
    }

    static func videoPosterQuery(subject: String) -> SemanticQuery {
        let normalizedSubject = normalizedWhitespace(subject)
        let searchText = normalizedSubject.lowercased()
        let visualSubject: String
        if containsAny(searchText, terms: ["游戏", "手游", "game", "gameplay"]) {
            visualSubject = "游戏截图"
        } else if containsAny(searchText, terms: ["录屏", "截图", "截屏", "screen", "screenshot"]) {
            visualSubject = "截图"
        } else {
            visualSubject = "\(normalizedSubject)图片"
        }
        return query(subject: visualSubject, searchText: searchText)
    }

    private static func query(subject: String, searchText: String) -> SemanticQuery {

        let wantsScreen = containsAny(searchText, terms: ["录屏", "截图", "截屏", "屏幕", "界面", "screen", "screenshot"])
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
            negativePrompts = ["其它动物照片", "狗的照片", "野生动物照片", "老虎照片", "风景照片", "人物照片", "手机截图"]
        } else if wantsDog {
            negativePrompts = ["猫的照片", "野生动物照片", "老虎照片", "风景照片", "人物照片", "手机截图"]
        } else if wantsScenery {
            negativePrompts = ["普通应用界面", "小猫", "小狗", "人像", "宠物", "文字文档"]
        } else if wantsPerson {
            negativePrompts = ["小猫", "小狗", "风景", "截图", "宠物"]
        } else {
            negativePrompts = ["与查询内容无关的图片", "随机图片", "模糊图片"]
        }

        return SemanticQuery(
            positivePrompts: orderedUnique([subject]),
            negativePrompts: orderedUnique(negativePrompts),
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
    let minimumSimilarity: Float
    let minimumMargin: Float
}

struct SemanticSearchOutcome {
    let candidates: [ResourceCandidate]
    let metrics: SemanticSearchMetrics
}

private struct SemanticAssetMatch {
    let asset: PHAsset
    let match: SemanticMatchScore
}

private struct NamedEmbedding: Codable {
    let name: String
    let values: [Float]
}

private enum ImageEmbeddingProfile: String, Codable, Equatable {
    case coarse
    case full

    var requestSide: Int {
        switch self {
        case .coarse:
            return 256
        case .full:
            return 512
        }
    }

    var expectedNames: Set<String> {
        ["full"]
    }

    var requestTimeout: TimeInterval {
        self == .coarse ? 1.0 : 3.0
    }
}

private final class PhotoImageRequestState {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<UIImage?, Never>?
    private var requestID = PHInvalidImageRequestID
    private weak var manager: PHImageManager?
    private var isCompleted = false

    func install(_ continuation: CheckedContinuation<UIImage?, Never>) -> Bool {
        lock.lock()
        if isCompleted {
            lock.unlock()
            continuation.resume(returning: nil)
            return false
        }
        self.continuation = continuation
        lock.unlock()
        return true
    }

    func bind(requestID: PHImageRequestID, manager: PHImageManager) {
        lock.lock()
        if isCompleted {
            lock.unlock()
            manager.cancelImageRequest(requestID)
            return
        }
        self.requestID = requestID
        self.manager = manager
        lock.unlock()
    }

    func complete(_ image: UIImage?) {
        let continuation: CheckedContinuation<UIImage?, Never>?
        lock.lock()
        guard !isCompleted else {
            lock.unlock()
            return
        }
        isCompleted = true
        continuation = self.continuation
        self.continuation = nil
        manager = nil
        lock.unlock()
        continuation?.resume(returning: image)
    }

    func cancel() {
        let continuation: CheckedContinuation<UIImage?, Never>?
        let manager: PHImageManager?
        let requestID: PHImageRequestID
        lock.lock()
        guard !isCompleted else {
            lock.unlock()
            return
        }
        isCompleted = true
        continuation = self.continuation
        self.continuation = nil
        manager = self.manager
        self.manager = nil
        requestID = self.requestID
        lock.unlock()

        if requestID != PHInvalidImageRequestID {
            manager?.cancelImageRequest(requestID)
        }
        continuation?.resume(returning: nil)
    }
}

private struct ImageEmbeddingCacheKey {
    let rawValue: String
    let profile: ImageEmbeddingProfile
    let allowsPersistence: Bool

    init(asset: PHAsset, profile: ImageEmbeddingProfile) {
        self.profile = profile
        let modifiedAt: String
        if let modificationDate = asset.modificationDate {
            modifiedAt = String(Int64(modificationDate.timeIntervalSince1970 * 1_000))
            allowsPersistence = true
        } else {
            modifiedAt = "session-only"
            allowsPersistence = false
        }
        rawValue = [
            asset.localIdentifier,
            modifiedAt,
            "\(asset.pixelWidth)x\(asset.pixelHeight)",
            profile.rawValue
        ].joined(separator: "|")
    }
}

private enum EmbeddingCacheLookup {
    case memory([NamedEmbedding])
    case disk([NamedEmbedding])
    case miss
    case corrupt
    case unavailable
}

private struct PersistentEmbeddingRecord: Codable {
    let cacheKey: String
    let profile: ImageEmbeddingProfile
    let embeddings: [NamedEmbedding]
}

private final class NamedEmbeddingBox: NSObject {
    let embeddings: [NamedEmbedding]

    init(_ embeddings: [NamedEmbedding]) {
        self.embeddings = embeddings
    }
}

private final class ImageEmbeddingStore {
    private let memory = NSCache<NSString, NamedEmbeddingBox>()
    private let ioQueue = DispatchQueue(label: "IntentResourceDemo.ImageEmbeddingStore")
    private let directoryURL: URL?

    init(namespace: String) {
        memory.countLimit = 512
        guard let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first else {
            directoryURL = nil
            return
        }

        var directory = applicationSupport
            .appendingPathComponent("IntentResourceDemo", isDirectory: true)
            .appendingPathComponent("SemanticImageEmbeddings", isDirectory: true)
            .appendingPathComponent(namespace, isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try directory.setResourceValues(values)
            directoryURL = directory
        } catch {
            directoryURL = nil
        }
    }

    func lookup(_ key: ImageEmbeddingCacheKey) -> EmbeddingCacheLookup {
        if let cached = memory.object(forKey: key.rawValue as NSString) {
            return .memory(cached.embeddings)
        }
        guard key.allowsPersistence else {
            return .miss
        }
        guard let directoryURL else {
            return .unavailable
        }

        let result: EmbeddingCacheLookup = ioQueue.sync {
            let url = fileURL(for: key, directoryURL: directoryURL)
            guard FileManager.default.fileExists(atPath: url.path) else {
                return .miss
            }
            do {
                let data = try Data(contentsOf: url, options: .mappedIfSafe)
                let record = try PropertyListDecoder().decode(PersistentEmbeddingRecord.self, from: data)
                guard record.cacheKey == key.rawValue,
                      record.profile == key.profile,
                      Self.isValid(record.embeddings, profile: key.profile) else {
                    Self.removeCorruptFile(at: url)
                    return .corrupt
                }
                return .disk(record.embeddings)
            } catch {
                Self.removeCorruptFile(at: url)
                return .corrupt
            }
        }
        if case .disk(let embeddings) = result {
            memory.setObject(NamedEmbeddingBox(embeddings), forKey: key.rawValue as NSString)
        }
        return result
    }

    func store(_ embeddings: [NamedEmbedding], for key: ImageEmbeddingCacheKey) {
        guard Self.isValid(embeddings, profile: key.profile) else { return }
        memory.setObject(NamedEmbeddingBox(embeddings), forKey: key.rawValue as NSString)
        guard key.allowsPersistence, let directoryURL else { return }

        let record = PersistentEmbeddingRecord(
            cacheKey: key.rawValue,
            profile: key.profile,
            embeddings: embeddings
        )
        ioQueue.async {
            do {
                let encoder = PropertyListEncoder()
                encoder.outputFormat = .binary
                let data = try encoder.encode(record)
                try data.write(
                    to: self.fileURL(for: key, directoryURL: directoryURL),
                    options: .atomic
                )
            } catch {
                NSLog("Chinese-CLIP embedding cache write failed: %@", error.localizedDescription)
            }
        }
    }

    private func fileURL(for key: ImageEmbeddingCacheKey, directoryURL: URL) -> URL {
        let digest = SHA256.hash(data: Data(key.rawValue.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
        return directoryURL.appendingPathComponent(digest).appendingPathExtension("plist")
    }

    private static func isValid(
        _ embeddings: [NamedEmbedding],
        profile: ImageEmbeddingProfile
    ) -> Bool {
        let names = embeddings.map(\.name)
        guard !embeddings.isEmpty,
              Set(names).count == names.count,
              Set(names).isSubset(of: profile.expectedNames),
              names == ["full"] else {
            return false
        }
        return embeddings.allSatisfy { embedding in
            guard embedding.values.count == ChineseCLIPModelContract.embeddingDimensions,
                  embedding.values.allSatisfy(\.isFinite) else {
                return false
            }
            let norm = sqrt(embedding.values.reduce(Float(0)) { $0 + $1 * $1 })
            return norm.isFinite && abs(norm - 1) < 0.01
        }
    }

    private static func removeCorruptFile(at url: URL) {
        do {
            try FileManager.default.removeItem(at: url)
        } catch {
            NSLog("Chinese-CLIP corrupt cache cleanup failed: %@", error.localizedDescription)
        }
    }
}

private final class SemanticSearchMetricsCollector {
    private let lock = NSLock()
    private var memoryCacheHits = 0
    private var diskCacheHits = 0
    private var cacheMisses = 0
    private var corruptCacheEvictions = 0
    private var cacheStorageUnavailable = 0
    private var imageRequestFailures = 0
    private var coarsePredictionCount = 0
    private var qualityPredictionCount = 0

    func recordMemoryHit() {
        withLock { memoryCacheHits += 1 }
    }

    func recordDiskHit() {
        withLock { diskCacheHits += 1 }
    }

    func recordCacheMiss() {
        withLock { cacheMisses += 1 }
    }

    func recordCorruptEviction() {
        withLock { corruptCacheEvictions += 1 }
    }

    func recordStorageUnavailable() {
        withLock { cacheStorageUnavailable += 1 }
    }

    func recordImageRequestFailure() {
        withLock { imageRequestFailures += 1 }
    }

    func recordPredictions(profile: ImageEmbeddingProfile, count: Int) {
        withLock {
            switch profile {
            case .coarse:
                coarsePredictionCount += count
            case .full:
                qualityPredictionCount += count
            }
        }
    }

    func snapshot(
        modelLoadTimeMs: Double,
        totalWallMs: Double,
        fullPassWallMs: Double,
        rerankWallMs: Double,
        scannedAssetCount: Int,
        shortlistAssetCount: Int
    ) -> SemanticSearchMetrics {
        lock.lock()
        defer { lock.unlock() }
        return SemanticSearchMetrics(
            modelLoadTimeMs: modelLoadTimeMs,
            totalWallMs: totalWallMs,
            fullPassWallMs: fullPassWallMs,
            rerankWallMs: rerankWallMs,
            scannedAssetCount: scannedAssetCount,
            shortlistAssetCount: shortlistAssetCount,
            memoryCacheHits: memoryCacheHits,
            diskCacheHits: diskCacheHits,
            cacheMisses: cacheMisses,
            corruptCacheEvictions: corruptCacheEvictions,
            cacheStorageUnavailable: cacheStorageUnavailable,
            imageRequestFailures: imageRequestFailures,
            coarsePredictionCount: coarsePredictionCount,
            qualityPredictionCount: qualityPredictionCount
        )
    }

    private func withLock(_ body: () -> Void) {
        lock.lock()
        body()
        lock.unlock()
    }
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

private extension CIImage {
    func translatedToOrigin() -> CIImage {
        transformed(
            by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y)
        )
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
