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
    private let predictionLock = NSLock()
    private var imageEmbeddingCache: [String: [NamedEmbedding]] = [:]
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
        semanticQueryText: String?,
        limit: Int
    ) async throws -> [ResourceCandidate] {
        guard isAvailable else {
            throw DemoError.resourceUnavailable("MobileCLIP semantic model is not loaded.")
        }

        let query = SemanticQueryPlanner.query(for: slots, semanticQueryText: semanticQueryText)
        guard let positiveEmbeddings = textEmbeddings(for: query.positivePrompts),
              let negativeEmbeddings = textEmbeddings(for: query.negativePrompts),
              !positiveEmbeddings.isEmpty,
              !negativeEmbeddings.isEmpty else {
            throw DemoError.resourceUnavailable("MobileCLIP text embeddings could not be generated.")
        }

        var candidates: [ResourceCandidate] = []
        let scanAssets = Array(assets.prefix(semanticScanLimit(for: slots)))
        let batchSize = semanticBatchSize(for: query)

        for batchStart in stride(from: 0, to: scanAssets.count, by: batchSize) {
            if Task.isCancelled { break }
            let batchEnd = min(batchStart + batchSize, scanAssets.count)
            let batch = Array(scanAssets[batchStart..<batchEnd])

            await withTaskGroup(of: ResourceCandidate?.self) { group in
                for asset in batch {
                    group.addTask {
                        await self.matchCandidate(
                            asset: asset,
                            kind: kind,
                            query: query,
                            positiveEmbeddings: positiveEmbeddings,
                            negativeEmbeddings: negativeEmbeddings
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
            return 3_000
        }
        if slots.resourcePhrase.contains("最近") || slots.qualifiers.selectionHint.contains("recent") {
            return 600
        }
        return 2_000
    }

    private func semanticBatchSize(for query: SemanticQuery) -> Int {
        2
    }

    private func matchCandidate(
        asset: PHAsset,
        kind: CandidateKind,
        query: SemanticQuery,
        positiveEmbeddings: [[Float]],
        negativeEmbeddings: [[Float]]
    ) async -> ResourceCandidate? {
        guard let imageEmbeddings = await imageEmbeddings(for: asset, query: query),
              !imageEmbeddings.isEmpty else {
            return nil
        }

        var bestPositive = SemanticMatchScore(value: -1, cropName: "none")
        var bestNegative: Float = -1

        for imageEmbedding in imageEmbeddings {
            let positive = positiveEmbeddings.map { cosineSimilarity(imageEmbedding.values, $0) }.max() ?? -1
            if positive > bestPositive.value {
                bestPositive = SemanticMatchScore(value: positive, cropName: imageEmbedding.name)
            }
            let negative = negativeEmbeddings.map { cosineSimilarity(imageEmbedding.values, $0) }.max() ?? -1
            bestNegative = max(bestNegative, negative)
        }

        let margin = bestPositive.value - bestNegative
        guard bestPositive.value >= query.minimumSimilarity,
              margin >= query.minimumMargin else {
            return nil
        }

        let score = Double(margin * 1000 + bestPositive.value * 10)
        return makeCandidate(
            asset: asset,
            kind: kind,
            score: score,
            positive: bestPositive.value,
            negative: bestNegative,
            margin: margin,
            cropName: bestPositive.cropName,
            prompts: query.positivePrompts
        )
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
            predictionLock.lock()
            defer { predictionLock.unlock() }
            let output = try textModel.prediction(text: inputArray).final_emb_1
            return normalized(Array(output.floatValues))
        } catch {
            return nil
        }
    }

    private func imageEmbeddings(for asset: PHAsset, query: SemanticQuery) async -> [NamedEmbedding]? {
        let cacheKey = "\(asset.localIdentifier)|\(query.embeddingProfile)"
        if let cached = cachedImageEmbeddings(cacheKey) {
            return cached
        }
        guard let image = await requestImage(asset: asset, needsRegionScan: query.needsRegionScan),
              let pixelBuffers = pixelBuffers(from: image, needsRegionScan: query.needsRegionScan),
              let imageModel else {
            return nil
        }

        var embeddings: [NamedEmbedding] = []
        for item in pixelBuffers {
            do {
                predictionLock.lock()
                let output = try imageModel.prediction(image: item.buffer).final_emb_1
                predictionLock.unlock()
                embeddings.append(NamedEmbedding(name: item.name, values: normalized(Array(output.floatValues))))
            } catch {
                predictionLock.unlock()
            }
        }

        guard !embeddings.isEmpty else {
            return nil
        }
        setCachedImageEmbeddings(embeddings, for: cacheKey)
        return embeddings
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
            options.deliveryMode = needsRegionScan ? .opportunistic : .fastFormat
            options.resizeMode = .fast
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

    private func pixelBuffers(from image: UIImage, needsRegionScan: Bool) -> [NamedPixelBuffer]? {
        guard let ciImage = CIImage(image: image) else {
            return nil
        }

        var views: [(String, CIImage)] = []
        if let full = ciImage.fitIntoSquare(size: CGSize(width: 256, height: 256)) {
            views.append(("full", full))
        }
        if let center = ciImage.cropToSquare(at: .center)?.resize(size: CGSize(width: 256, height: 256)) {
            views.append(("center", center))
        }
        if needsRegionScan {
            for anchor in RegionAnchor.cornerAnchors {
                if let crop = ciImage.cropToSquare(at: anchor)?.resize(size: CGSize(width: 256, height: 256)) {
                    views.append((anchor.rawValue, crop))
                }
            }
        }

        var result: [NamedPixelBuffer] = []
        var seen = Set<String>()
        for view in views where seen.insert(view.0).inserted {
            guard let buffer = pixelBuffer(from: view.1) else {
                continue
            }
            result.append(NamedPixelBuffer(name: view.0, buffer: buffer))
        }
        return result
    }

    private func pixelBuffer(from image: CIImage) -> CVPixelBuffer? {
        let translated = image.transformed(
            by: CGAffineTransform(translationX: -image.extent.origin.x, y: -image.extent.origin.y)
        )

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
        ciContext.render(translated, to: buffer)
        return buffer
    }

    private func makeCandidate(
        asset: PHAsset,
        kind: CandidateKind,
        score: Double,
        positive: Float,
        negative: Float,
        margin: Float,
        cropName: String,
        prompts: [String]
    ) -> ResourceCandidate {
        let date = asset.creationDate.map { DateFormatter.localizedString(from: $0, dateStyle: .short, timeStyle: .short) } ?? "未知时间"
        let dimensions = "\(asset.pixelWidth)x\(asset.pixelHeight)"
        let title = kind == .video ? "视频 \(date)" : "照片 \(date)"
        let promptPreview = prompts.prefix(4).joined(separator: " / ")
        let detail = String(
            format: "semantic margin %.3f, positive %.3f, negative %.3f, view %@, prompts: %@",
            Double(margin),
            Double(positive),
            Double(negative),
            cropName,
            promptPreview
        )

        return ResourceCandidate(
            id: asset.localIdentifier,
            kind: kind,
            title: title,
            subtitle: dimensions,
            detail: detail,
            score: score,
            debugInfo: "matched calibrated MobileCLIP semantic retrieval"
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

private enum SemanticQueryPlanner {
    static func query(for slots: NormalizedSlots, semanticQueryText: String?) -> SemanticQuery {
        let translated = semanticQueryText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let fallbackEnglish = englishTerms(
            from: [
                slots.searchKeyword ?? "",
                slots.resourcePhrase
            ]
            .joined(separator: " ")
            .lowercased()
        )
        let queryText = translated.isEmpty ? fallbackEnglish.joined(separator: " ") : translated
        let normalizedQuery = queryText
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let subject = normalizedQuery.isEmpty ? "the requested visual content" : normalizedQuery
        let lowerSubject = subject.lowercased()
        let wantsScreen = containsAny(
            lowerSubject,
            terms: ["screenshot", "screen capture", "screen", "interface", "gameplay", "game"]
        )
        let wantsGame = containsAny(lowerSubject, terms: ["game", "gameplay"])
        let wantsScenery = containsAny(
            lowerSubject,
            terms: ["landscape", "scenery", "nature", "outdoor", "mountain", "lake", "beach"]
        )
        let wantsWildlife = containsAny(lowerSubject, terms: ["wildlife", "wild animal", "zoo", "safari"])

        var positivePrompts = [
            subject,
            "an image matching \(subject)",
            "a photo containing \(subject)",
            "\(subject) visible anywhere in the image",
            "a close-up or background view containing \(subject)"
        ]
        positivePrompts.append(
            wantsScreen
            ? "a screenshot or app interface matching \(subject)"
            : "a real camera photo matching \(subject)"
        )
        if wantsGame {
            positivePrompts.append(contentsOf: [
                "a video game screenshot",
                "a gameplay screen",
                "a game interface"
            ])
        }
        if !wantsWildlife && !wantsScreen {
            positivePrompts.append("a domestic or everyday subject matching \(subject)")
        }

        var negativePrompts = [
            "an unrelated image",
            "a visually similar but wrong image",
            "a different object, animal, or scene",
            "a random photo",
            "a generic image that does not match the request",
            "a blurry or ambiguous image",
            "an image without \(subject)"
        ]
        if !wantsWildlife {
            negativePrompts.append(contentsOf: [
                "a wildlife animal photo",
                "a zoo animal photo",
                "a wild predator animal photo"
            ])
        }
        if wantsScreen {
            negativePrompts.append(contentsOf: [
                "a real camera photo",
                "a pet or animal photo",
                "a landscape camera photo"
            ])
        } else {
            negativePrompts.append("a screenshot or app interface that does not match the query")
        }
        if wantsGame {
            negativePrompts.append(contentsOf: [
                "a generic app screenshot",
                "a web app interface screenshot",
                "a phone settings screenshot"
            ])
        }

        return SemanticQuery(
            positivePrompts: orderedUnique(positivePrompts),
            negativePrompts: orderedUnique(negativePrompts),
            needsRegionScan: true,
            minimumSimilarity: 0.19,
            minimumMargin: wantsGame || wantsScenery ? 0.035 : 0.008
        )
    }

    private static func englishTerms(from text: String) -> [String] {
        let pattern = "[a-z0-9][a-z0-9 -]{1,80}"
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
        needsRegionScan ? "regions-v2" : "full-v2"
    }
}

private struct NamedEmbedding {
    let name: String
    let values: [Float]
}

private struct NamedPixelBuffer {
    let name: String
    let buffer: CVPixelBuffer
}

private struct SemanticMatchScore {
    let value: Float
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
    func cropToSquare(at anchor: RegionAnchor) -> CIImage? {
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
        return cropped(to: rect).transformed(by: CGAffineTransform(translationX: -x, y: -y))
    }

    func fitIntoSquare(size targetSize: CGSize) -> CIImage? {
        guard extent.width > 0, extent.height > 0 else {
            return nil
        }

        let scale = min(targetSize.width / extent.width, targetSize.height / extent.height)
        let scaledWidth = extent.width * scale
        let scaledHeight = extent.height * scale
        let x = (targetSize.width - scaledWidth) / 2
        let y = (targetSize.height - scaledHeight) / 2
        let scaled = transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: x, y: y))

        let background = CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5))
            .cropped(to: CGRect(origin: .zero, size: targetSize))
        return scaled.composited(over: background)
            .cropped(to: CGRect(origin: .zero, size: targetSize))
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
