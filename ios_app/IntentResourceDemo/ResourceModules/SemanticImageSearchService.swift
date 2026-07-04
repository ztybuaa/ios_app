import CoreImage
import CoreML
import Foundation
import Photos
import UIKit

final class SemanticImageSearchService {
    static let shared = SemanticImageSearchService()

    private let ciContext = CIContext()
    private let tokenizer: CLIPTokenizer?
    private let imageModel: mobileclip_s0_image?
    private let textModel: mobileclip_s0_text?
    private let cacheLock = NSLock()
    private var imageEmbeddingCache: [String: [Float]] = [:]
    private var textEmbeddingCache: [String: [Float]] = [:]

    private init() {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        tokenizer = CLIPTokenizer()
        imageModel = try? mobileclip_s0_image(configuration: configuration)
        textModel = try? mobileclip_s0_text(configuration: configuration)
    }

    var isAvailable: Bool {
        tokenizer != nil && imageModel != nil && textModel != nil
    }

    func search(
        assets: [PHAsset],
        kind: CandidateKind,
        slots: NormalizedSlots,
        limit: Int
    ) async throws -> [ResourceCandidate] {
        guard isAvailable else {
            throw DemoError.resourceUnavailable("MobileCLIP 语义检索模型未加载，无法执行自然语言图片检索。")
        }

        let prompts = SemanticPromptBuilder.prompts(for: slots)
        guard !prompts.isEmpty,
              let textEmbeddings = textEmbeddings(for: prompts),
              !textEmbeddings.isEmpty else {
            throw DemoError.resourceUnavailable("MobileCLIP 文本向量生成失败，无法执行语义图片检索。")
        }

        var candidates: [ResourceCandidate] = []
        let scanAssets = Array(assets.prefix(semanticScanLimit(for: slots)))

        for batchStart in stride(from: 0, to: scanAssets.count, by: semanticBatchSize(for: slots)) {
            if Task.isCancelled { break }
            let batchEnd = min(batchStart + semanticBatchSize(for: slots), scanAssets.count)
            let batch = Array(scanAssets[batchStart..<batchEnd])

            await withTaskGroup(of: ResourceCandidate?.self) { group in
                for asset in batch {
                    group.addTask {
                        await self.matchCandidate(
                            asset: asset,
                            kind: kind,
                            textEmbeddings: textEmbeddings,
                            prompts: prompts,
                            threshold: self.semanticThreshold(for: slots)
                        )
                    }
                }

                for await candidate in group {
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

    private func semanticScanLimit(for slots: NormalizedSlots) -> Int {
        if slots.qualifiers.selectionHint.contains("all") {
            return 700
        }
        if slots.resourcePhrase.contains("最近") || slots.qualifiers.selectionHint.contains("recent") {
            return 240
        }
        return 420
    }

    private func semanticBatchSize(for slots: NormalizedSlots) -> Int {
        if slots.qualifiers.selectionHint.contains("all") {
            return 3
        }
        return 4
    }

    private func semanticThreshold(for slots: NormalizedSlots) -> Float {
        let text = "\(slots.searchKeyword ?? "") \(slots.resourcePhrase)"
        if text.contains("猫") || text.contains("狗") {
            return 0.18
        }
        if text.contains("游戏") || text.contains("截图") {
            return 0.16
        }
        return 0.17
    }

    private func matchCandidate(
        asset: PHAsset,
        kind: CandidateKind,
        textEmbeddings: [[Float]],
        prompts: [String],
        threshold: Float
    ) async -> ResourceCandidate? {
        guard let imageEmbedding = await imageEmbedding(for: asset) else {
            return nil
        }

        let best = textEmbeddings
            .map { cosineSimilarity(imageEmbedding, $0) }
            .max() ?? -1
        guard best >= threshold else {
            return nil
        }
        return makeCandidate(asset: asset, kind: kind, score: Double(best), prompts: prompts)
    }

    private func textEmbeddings(for prompts: [String]) -> [[Float]]? {
        var embeddings: [[Float]] = []
        for prompt in prompts {
            if let cached = cachedTextEmbedding(prompt) {
                embeddings.append(cached)
                continue
            }
            guard let embedding = computeTextEmbedding(prompt) else {
                continue
            }
            setCachedTextEmbedding(embedding, for: prompt)
            embeddings.append(embedding)
        }
        return embeddings
    }

    private func computeTextEmbedding(_ prompt: String) -> [Float]? {
        guard let tokenizer, let textModel else {
            return nil
        }
        do {
            let inputIds = tokenizer.encode_full(text: prompt)
            let inputArray = try MLMultiArray(shape: [1, 77], dataType: .int32)
            for index in 0..<min(inputIds.count, 77) {
                inputArray[index] = NSNumber(value: inputIds[index])
            }
            let output = try textModel.prediction(text: inputArray).final_emb_1
            return normalized(Array(output.floatValues))
        } catch {
            return nil
        }
    }

    private func imageEmbedding(for asset: PHAsset) async -> [Float]? {
        if let cached = cachedImageEmbedding(asset.localIdentifier) {
            return cached
        }
        guard let image = await requestImage(asset: asset),
              let pixelBuffer = pixelBuffer(from: image),
              let imageModel else {
            return nil
        }

        do {
            let output = try imageModel.prediction(image: pixelBuffer).final_emb_1
            let embedding = normalized(Array(output.floatValues))
            setCachedImageEmbedding(embedding, for: asset.localIdentifier)
            return embedding
        } catch {
            return nil
        }
    }

    private func requestImage(asset: PHAsset) async -> UIImage? {
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
            options.deliveryMode = .fastFormat
            options.resizeMode = .fast
            options.isNetworkAccessAllowed = false

            PHImageManager.default().requestImage(
                for: asset,
                targetSize: CGSize(width: 256, height: 256),
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

    private func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        guard var ciImage = CIImage(image: image)?.cropToSquare()?.resize(size: CGSize(width: 256, height: 256)) else {
            return nil
        }
        ciImage = ciImage.transformed(by: CGAffineTransform(translationX: -ciImage.extent.origin.x, y: -ciImage.extent.origin.y))

        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(
            nil,
            256,
            256,
            kCVPixelFormatType_32ARGB,
            nil,
            &buffer
        )
        guard let buffer else {
            return nil
        }
        ciContext.render(ciImage, to: buffer)
        return buffer
    }

    private func makeCandidate(asset: PHAsset, kind: CandidateKind, score: Double, prompts: [String]) -> ResourceCandidate {
        let date = asset.creationDate.map { DateFormatter.localizedString(from: $0, dateStyle: .short, timeStyle: .short) } ?? "未知时间"
        let dimensions = "\(asset.pixelWidth)x\(asset.pixelHeight)"
        let title = kind == .video ? "视频 \(date)" : "照片 \(date)"
        let promptPreview = prompts.prefix(4).joined(separator: " / ")

        return ResourceCandidate(
            id: asset.localIdentifier,
            kind: kind,
            title: title,
            subtitle: dimensions,
            detail: "semantic prompts: \(promptPreview)",
            score: score * 100,
            debugInfo: "matched MobileCLIP semantic similarity"
        )
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else {
            return -1
        }
        return zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
    }

    private func normalized(_ values: [Float]) -> [Float] {
        let norm = sqrt(values.reduce(Float(0)) { $0 + $1 * $1 })
        guard norm > 0 else {
            return values
        }
        return values.map { $0 / norm }
    }

    private func cachedImageEmbedding(_ key: String) -> [Float]? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return imageEmbeddingCache[key]
    }

    private func setCachedImageEmbedding(_ value: [Float], for key: String) {
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

private enum SemanticPromptBuilder {
    private static let conceptMap: [(String, [String])] = [
        ("猫", ["cat", "kitten", "domestic cat"]),
        ("狗", ["dog", "puppy"]),
        ("宠物", ["pet animal"]),
        ("动物", ["animal"]),
        ("鸟", ["bird"]),
        ("鱼", ["fish"]),
        ("花", ["flower"]),
        ("树", ["tree"]),
        ("草地", ["grass field"]),
        ("天空", ["sky"]),
        ("云", ["clouds"]),
        ("雪", ["snow"]),
        ("海", ["sea", "ocean"]),
        ("山", ["mountain"]),
        ("河", ["river"]),
        ("湖", ["lake"]),
        ("风景", ["landscape", "scenery"]),
        ("旅行", ["travel photo"]),
        ("街景", ["street scene"]),
        ("夜景", ["night scene"]),
        ("人像", ["portrait photo of a person"]),
        ("自拍", ["selfie"]),
        ("合影", ["group photo"]),
        ("小孩", ["child"]),
        ("孩子", ["child"]),
        ("老人", ["elderly person"]),
        ("人物", ["person"]),
        ("人", ["person"]),
        ("游戏截图", ["video game screenshot", "gameplay screen"]),
        ("游戏", ["video game", "gameplay"]),
        ("截图", ["screenshot", "phone screen capture"]),
        ("截屏", ["screenshot", "phone screen capture"]),
        ("聊天", ["chat screenshot", "messaging app screen"]),
        ("微信", ["WeChat screenshot", "chat app screen"]),
        ("代码", ["code editor screenshot", "programming code"]),
        ("网页", ["web page screenshot"]),
        ("App", ["mobile app interface"]),
        ("应用", ["mobile app interface"]),
        ("地图", ["map screenshot"]),
        ("票", ["ticket", "receipt"]),
        ("证件", ["identity document", "card document"]),
        ("文档", ["document screenshot"]),
        ("表格", ["spreadsheet", "table document"]),
        ("幻灯片", ["presentation slide"]),
        ("PPT", ["presentation slide"]),
        ("博物馆", ["museum exhibition"]),
        ("展览", ["exhibition hall"]),
        ("会议", ["meeting", "conference room"]),
        ("课堂", ["classroom"]),
        ("白板", ["whiteboard"]),
        ("电脑", ["computer"]),
        ("手机", ["phone"]),
        ("车", ["car"]),
        ("自行车", ["bicycle"]),
        ("食物", ["food"]),
        ("饭", ["meal", "food"]),
        ("咖啡", ["coffee"]),
        ("蛋糕", ["cake"]),
        ("衣服", ["clothing"]),
        ("鞋", ["shoes"]),
        ("包", ["bag"]),
        ("红色", ["red"]),
        ("蓝色", ["blue"]),
        ("绿色", ["green"]),
        ("黄色", ["yellow"]),
        ("黑色", ["black"]),
        ("白色", ["white"])
    ]

    static func prompts(for slots: NormalizedSlots) -> [String] {
        let phrase = slots.resourcePhrase.trimmingCharacters(in: .whitespacesAndNewlines)
        let keyword = slots.searchKeyword?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let query = orderedUnique([keyword, phrase]).joined(separator: " ")
        let lowerQuery = query.lowercased()

        var concepts: [String] = []
        for (source, translations) in conceptMap where query.contains(source) {
            concepts.append(contentsOf: translations)
        }

        concepts.append(contentsOf: englishTerms(from: lowerQuery))
        concepts = orderedUnique(concepts)

        var prompts: [String] = []
        if concepts.isEmpty {
            prompts.append(query)
            prompts.append("an image matching the user request")
            prompts.append("a photo related to the query")
        } else {
            let combined = concepts.prefix(4).joined(separator: ", ")
            prompts.append("an image containing \(combined)")
            prompts.append("a photo containing \(combined)")
            prompts.append("\(combined) visible in the image")

            for concept in concepts.prefix(8) {
                prompts.append("a photo containing \(concept)")
                prompts.append("\(concept) visible in the image")
            }
        }

        if query.contains("截图") || query.contains("截屏") || lowerQuery.contains("screenshot") {
            prompts.append("a screenshot matching \(concepts.first ?? "the query")")
            prompts.append("a phone screen capture with \(concepts.prefix(3).joined(separator: ", "))")
        }

        if query.contains("背景") || query.contains("角落") || query.contains("旁边") {
            for concept in concepts.prefix(4) {
                prompts.append("\(concept) in the background")
                prompts.append("a small \(concept) visible in the image")
            }
        }

        if !keyword.isEmpty {
            prompts.append(keyword)
        }
        if !phrase.isEmpty {
            prompts.append(phrase)
        }
        return orderedUnique(prompts.filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
    }

    private static func englishTerms(from text: String) -> [String] {
        let pattern = "[a-z0-9][a-z0-9 -]{1,40}"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return []
        }
        let range = NSRange(location: 0, length: text.utf16.count)
        return regex.matches(in: text, options: [], range: range).compactMap { match in
            guard let swiftRange = Range(match.range, in: text) else { return nil }
            return String(text[swiftRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private static func orderedUnique(_ values: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for value in values {
            let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !normalized.isEmpty else { continue }
            if seen.insert(normalized.lowercased()).inserted {
                result.append(normalized)
            }
        }
        return result
    }
}

private extension CIImage {
    func cropToSquare() -> CIImage? {
        let size = min(extent.width, extent.height)
        let x = round((extent.width - size) / 2)
        let y = round((extent.height - size) / 2)
        let rect = CGRect(x: x, y: y, width: size, height: size)
        return cropped(to: rect).transformed(by: CGAffineTransform(translationX: -x, y: -y))
    }

    func resize(size: CGSize) -> CIImage? {
        let scaleX = size.width / extent.width
        let scaleY = size.height / extent.height
        return transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }
}

private extension MLMultiArray {
    var floatValues: [Float] {
        let count = self.count
        var values: [Float] = []
        values.reserveCapacity(count)
        for index in 0..<count {
            values.append(self[index].floatValue)
        }
        return values
    }
}
