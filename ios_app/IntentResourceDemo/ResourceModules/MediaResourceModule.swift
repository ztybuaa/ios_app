import Foundation
import Photos
import UIKit
import Vision

final class MediaResourceModule {
    func search(kind: CandidateKind, slots: NormalizedSlots, limit: Int = 12) async throws -> [ResourceCandidate] {
        try await ensureAccess()

        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        fetchOptions.fetchLimit = scanLimit(for: slots, resultLimit: limit)
        fetchOptions.predicate = NSPredicate(
            format: "mediaType == %d",
            kind == .video ? PHAssetMediaType.video.rawValue : PHAssetMediaType.image.rawValue
        )

        let assets = PHAsset.fetchAssets(with: fetchOptions)
        var candidates: [ResourceCandidate] = []

        for index in 0..<assets.count {
            if Task.isCancelled { break }
            let asset = assets.object(at: index)
            let vision = await visionResult(for: asset)
            let score = score(asset: asset, vision: vision, slots: slots)
            if shouldIncludeCandidate(score: score, slots: slots) {
                candidates.append(makeCandidate(asset: asset, kind: kind, vision: vision, score: score))
            }
        }

        return candidates
            .sorted { $0.score == $1.score ? $0.title < $1.title : $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }

    private func ensureAccess() async throws {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        switch status {
        case .authorized, .limited:
            return
        case .notDetermined:
            let newStatus = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
            if newStatus == .authorized || newStatus == .limited { return }
            throw DemoError.permissionDenied("相册权限被拒绝，无法检索照片或视频候选。")
        case .denied, .restricted:
            throw DemoError.permissionDenied("相册权限不可用，请在系统设置中允许访问照片。")
        @unknown default:
            throw DemoError.permissionDenied("相册权限状态未知，无法继续检索。")
        }
    }

    private func scanLimit(for slots: NormalizedSlots, resultLimit: Int) -> Int {
        if slots.searchKeyword == nil && slots.resourcePhrase.isEmpty {
            return resultLimit
        }
        return max(resultLimit, 500)
    }

    private func shouldIncludeCandidate(score: Double, slots: NormalizedSlots) -> Bool {
        if score > 0 {
            return true
        }
        let keyword = slots.searchKeyword?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let hasSearchTerm = !keyword.isEmpty || !slots.resourcePhrase.isEmpty
        if hasSearchTerm {
            return false
        }
        return slots.qualifiers.selectionHint.contains("recent")
    }

    private func score(asset: PHAsset, vision: AssetVisionResult, slots: NormalizedSlots) -> Double {
        if let animalQuery = animalQuery(for: slots) {
            guard let detection = bestAnimalDetection(for: animalQuery, in: vision.animals) else {
                return 0
            }

            var score = 8.0 + Double(detection.confidence) * 12.0
            score += min(detection.area * 20.0, 3.0)
            if slots.qualifiers.selectionHint.contains("recent") {
                score += 1.0
            }
            if slots.qualifiers.time.contains("今天"), Calendar.current.isDateInToday(asset.creationDate ?? .distantPast) {
                score += 1.0
            }
            return score
        }

        let keywordAliases = aliases(for: slots)
        let labelText = vision.labels.joined(separator: " ")
        var score = CandidateScorer.score(
            keyword: slots.searchKeyword,
            phrase: slots.resourcePhrase,
            targetText: labelText,
            tags: [],
            qualifiers: slots.qualifiers
        )

        for alias in keywordAliases where containsLabel(alias, in: vision.labels) {
            score += alias.count <= 3 ? 4.0 : 3.0
        }
        if slots.qualifiers.selectionHint.contains("recent") {
            score += 1.0
        }
        if slots.qualifiers.time.contains("今天"), Calendar.current.isDateInToday(asset.creationDate ?? .distantPast) {
            score += 1.0
        }
        if slots.resourcePhrase.contains("截图"), asset.mediaSubtypes.contains(.photoScreenshot) {
            score += 3.0
        }

        return score
    }

    private func makeCandidate(asset: PHAsset, kind: CandidateKind, vision: AssetVisionResult, score: Double) -> ResourceCandidate {
        let date = asset.creationDate.map { DateFormatter.localizedString(from: $0, dateStyle: .short, timeStyle: .short) } ?? "未知时间"
        let dimensions = "\(asset.pixelWidth)x\(asset.pixelHeight)"
        let title = kind == .video ? "视频 \(date)" : "照片 \(date)"

        return ResourceCandidate(
            id: asset.localIdentifier,
            kind: kind,
            title: title,
            subtitle: dimensions,
            detail: detailText(for: vision),
            score: score,
            debugInfo: "matched Photos metadata + Vision animal/object detection; sequential scan"
        )
    }

    private func detailText(for vision: AssetVisionResult) -> String {
        let animalText = vision.animals
            .prefix(6)
            .map { "\($0.label) \(String(format: "%.2f", $0.confidence))" }
        let labels = vision.labels.prefix(8)
        let detail = (animalText + labels).joined(separator: ", ")
        return detail.isEmpty ? "无视觉标签" : detail
    }

    private func visionResult(for asset: PHAsset) async -> AssetVisionResult {
        let metadataLabels = metadataLabels(for: asset)
        guard let image = await requestThumbnail(asset: asset), let cgImage = image.cgImage else {
            return AssetVisionResult(labels: uniqueLowercased(metadataLabels), animals: [])
        }

        let result = await analyzeImage(cgImage)
        return AssetVisionResult(
            labels: uniqueLowercased(metadataLabels + result.labels),
            animals: result.animals
        )
    }

    private func metadataLabels(for asset: PHAsset) -> [String] {
        var labels: [String] = []
        if asset.mediaSubtypes.contains(.photoScreenshot) {
            labels.append("screenshot")
            labels.append("截图")
        }
        if asset.isFavorite {
            labels.append("favorite")
            labels.append("精选")
        }
        if let creationDate = asset.creationDate {
            if Calendar.current.isDateInToday(creationDate) {
                labels.append("today")
                labels.append("今天")
            }
            if Calendar.current.isDateInYesterday(creationDate) {
                labels.append("yesterday")
                labels.append("昨天")
            }
        }
        labels.append(asset.mediaType == .video ? "video" : "photo")
        labels.append(asset.mediaType == .video ? "视频" : "照片")
        return labels
    }

    private func requestThumbnail(asset: PHAsset) async -> UIImage? {
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
            options.resizeMode = .fast
            options.isNetworkAccessAllowed = false

            PHImageManager.default().requestImage(
                for: asset,
                targetSize: CGSize(width: 1024, height: 1024),
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

    private func analyzeImage(_ cgImage: CGImage) async -> AssetVisionResult {
        let task = Task.detached(priority: .utility) {
            var labels: [String] = []
            var animals: [AnimalDetection] = []

            let animalRequest = VNRecognizeAnimalsRequest()
            let classifyRequest = VNClassifyImageRequest()

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([animalRequest, classifyRequest])
                for observation in animalRequest.results ?? [] {
                    let area = Double(observation.boundingBox.width * observation.boundingBox.height)
                    for label in observation.labels {
                        let normalized = label.identifier.lowercased()
                        animals.append(
                            AnimalDetection(
                                label: normalized,
                                confidence: label.confidence,
                                area: area
                            )
                        )
                        labels.append(normalized)
                    }
                }
                labels.append(
                    contentsOf: (classifyRequest.results ?? [])
                        .filter { $0.confidence >= 0.65 }
                        .prefix(12)
                        .map { $0.identifier.lowercased() }
                )
            } catch {
                return AssetVisionResult(labels: labels, animals: animals)
            }

            return AssetVisionResult(
                labels: labels.flatMap(Self.expandVisionIdentifier),
                animals: animals
            )
        }
        return await task.value
    }

    private static func expandVisionIdentifier(_ identifier: String) -> [String] {
        let cleaned = identifier
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: ",", with: " ")
            .lowercased()
        let pieces = cleaned
            .split(separator: " ")
            .map(String.init)
        return Array(Set([cleaned] + pieces))
    }

    private func uniqueLowercased(_ labels: [String]) -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for label in labels {
            let normalized = label.lowercased()
            if seen.insert(normalized).inserted {
                result.append(normalized)
            }
        }
        return result
    }

    private func animalQuery(for slots: NormalizedSlots) -> AnimalQuery? {
        let text = "\(slots.searchKeyword ?? "") \(slots.resourcePhrase)"
        if text.contains("猫") {
            return AnimalQuery(
                aliases: [
                    "cat",
                    "kitten",
                    "feline",
                    "tabby",
                    "tabby cat",
                    "tiger cat",
                    "egyptian cat",
                    "persian cat",
                    "siamese cat",
                    "lynx"
                ],
                minimumConfidence: 0.12
            )
        }
        if text.contains("狗") {
            return AnimalQuery(
                aliases: ["dog", "puppy", "canine"],
                minimumConfidence: 0.18
            )
        }
        return nil
    }

    private func bestAnimalDetection(for query: AnimalQuery, in detections: [AnimalDetection]) -> AnimalDetection? {
        detections
            .filter { detection in
                detection.confidence >= query.minimumConfidence &&
                    query.aliases.contains { labelMatches(alias: $0, label: detection.label) }
            }
            .sorted { lhs, rhs in
                if lhs.confidence == rhs.confidence {
                    return lhs.area > rhs.area
                }
                return lhs.confidence > rhs.confidence
            }
            .first
    }

    private func containsLabel(_ alias: String, in labels: [String]) -> Bool {
        let normalizedAlias = alias.lowercased()
        return labels.contains { labelMatches(alias: normalizedAlias, label: $0) }
    }

    private func labelMatches(alias: String, label: String) -> Bool {
        let normalizedLabel = label.lowercased()
        if alias.contains(" ") {
            return normalizedLabel == alias
        }

        let tokens = normalizedLabel
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: ",", with: " ")
            .split(separator: " ")
            .map(String.init)
        return tokens.contains(alias)
    }

    private func aliases(for slots: NormalizedSlots) -> [String] {
        guard let keyword = slots.searchKeyword else { return [] }
        let table: [String: [String]] = [
            "人": ["person", "people", "human"],
            "会议": ["meeting", "conference", "presentation"],
            "旅行": ["travel", "landscape", "outdoor"],
            "风景": ["landscape", "sky", "mountain", "beach"],
            "截图": ["screenshot", "screen"]
        ]
        return table[keyword] ?? [keyword.lowercased()]
    }
}

private struct AssetVisionResult {
    let labels: [String]
    let animals: [AnimalDetection]
}

private struct AnimalDetection {
    let label: String
    let confidence: Float
    let area: Double
}

private struct AnimalQuery {
    let aliases: [String]
    let minimumConfidence: Float
}
