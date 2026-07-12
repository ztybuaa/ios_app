import Foundation
import Photos
import UIKit
import Vision

final class MediaResourceModule {
    func search(
        kind: CandidateKind,
        slots: NormalizedSlots,
        limit: Int = 12
    ) async throws -> MediaSearchOutcome {
        try await ensureAccess()

        let plan = MediaQueryPlan(slots: slots)
        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        fetchOptions.fetchLimit = mediaFetchLimit(kind: kind, plan: plan, resultLimit: limit)
        fetchOptions.predicate = NSPredicate(
            format: "mediaType == %d",
            kind == .video ? PHAssetMediaType.video.rawValue : PHAssetMediaType.image.rawValue
        )

        let assets = PHAsset.fetchAssets(with: fetchOptions)
        let assetList = (0..<assets.count).map { assets.object(at: $0) }
        if kind == .photo,
           plan.hasSearchTerm {
            let semanticOutcome = try await SemanticImageSearchService.shared.search(
                assets: assetList,
                kind: kind,
                slots: slots,
                limit: limit
            )
            return MediaSearchOutcome(
                candidates: semanticOutcome.candidates,
                semanticMetrics: semanticOutcome.metrics
            )
        }

        var candidates: [ResourceCandidate] = []

        for asset in assetList {
            if Task.isCancelled { break }
            let vision = await visionResult(for: asset, plan: plan)
            let score = score(asset: asset, vision: vision, plan: plan)
            if shouldIncludeCandidate(score: score, plan: plan) {
                candidates.append(makeCandidate(asset: asset, kind: kind, vision: vision, score: score))
            }
        }

        return MediaSearchOutcome(
            candidates: candidates
                .sorted { $0.score == $1.score ? $0.title < $1.title : $0.score > $1.score }
                .prefix(limit)
                .map { $0 },
            semanticMetrics: nil
        )
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

    private func scanLimit(for plan: MediaQueryPlan, resultLimit: Int) -> Int {
        if !plan.hasSearchTerm {
            return resultLimit
        }
        return plan.needsFineVisualSearch ? max(resultLimit, 500) : max(resultLimit, 240)
    }

    private func mediaFetchLimit(kind: CandidateKind, plan: MediaQueryPlan, resultLimit: Int) -> Int {
        if kind == .photo && plan.hasSearchTerm {
            if plan.wantsRecent {
                return 600
            }
            return 2_500
        }
        return scanLimit(for: plan, resultLimit: resultLimit)
    }

    private func shouldIncludeCandidate(score: Double, plan: MediaQueryPlan) -> Bool {
        if score > 0 {
            return true
        }
        return !plan.hasSearchTerm && plan.wantsRecent
    }

    private func score(asset: PHAsset, vision: AssetVisionResult, plan: MediaQueryPlan) -> Double {
        var score = 0.0

        if plan.wantsScreenshot {
            if asset.mediaSubtypes.contains(.photoScreenshot) || vision.labels.contains("screenshot") {
                score += 5.0
            } else {
                return 0
            }
        }

        if let animalQuery = plan.animalQuery {
            guard let detection = bestAnimalDetection(for: animalQuery, in: vision.animals) else {
                return 0
            }
            score += 8.0 + Double(detection.confidence) * 12.0
            score += min(detection.area * 20.0, 3.0)
        }

        if plan.wantsHuman {
            let humanScore = humanEvidenceScore(vision)
            guard humanScore > 0 else {
                return 0
            }
            score += humanScore
        }

        if plan.wantsGame {
            let gameScore = gameEvidenceScore(vision)
            guard gameScore > 0 else {
                return 0
            }
            score += gameScore
        }

        let labelScore = semanticLabelScore(aliases: plan.semanticAliases, labels: vision.labels)
        let textScore = textScore(aliases: plan.semanticAliases, textLines: vision.textLines)
        score += labelScore + textScore

        if plan.hasSearchTerm && score == 0 {
            return 0
        }
        if plan.wantsRecent {
            score += 1.0
        }
        if plan.wantsToday, Calendar.current.isDateInToday(asset.creationDate ?? .distantPast) {
            score += 1.0
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
            debugInfo: "matched query-planned Vision evidence: animal/human/text/classification/metadata"
        )
    }

    private func detailText(for vision: AssetVisionResult) -> String {
        let animals = vision.animals
            .prefix(4)
            .map { "\($0.label) \(String(format: "%.2f", $0.confidence))" }
        let humans = vision.humans
            .prefix(3)
            .map { "human \(String(format: "%.2f", $0.confidence))" }
        let faces = vision.faces
            .prefix(3)
            .map { "face \(String(format: "%.2f", $0.confidence))" }
        let labels = vision.labels.prefix(8)
        let text = vision.textLines.prefix(4).map { "OCR:\($0)" }
        let detail = (animals + humans + faces + labels + text).joined(separator: ", ")
        return detail.isEmpty ? "无视觉证据" : detail
    }

    private func visionResult(for asset: PHAsset, plan: MediaQueryPlan) async -> AssetVisionResult {
        let metadataLabels = metadataLabels(for: asset)
        guard plan.needsVision,
              let image = await requestThumbnail(asset: asset, plan: plan),
              let cgImage = image.cgImage else {
            return AssetVisionResult(labels: uniqueLowercased(metadataLabels), animals: [], humans: [], faces: [], textLines: [])
        }

        let result = await analyzeImage(cgImage, plan: plan)
        return AssetVisionResult(
            labels: uniqueLowercased(metadataLabels + result.labels),
            animals: result.animals,
            humans: result.humans,
            faces: result.faces,
            textLines: uniqueLowercased(result.textLines)
        )
    }

    private func metadataLabels(for asset: PHAsset) -> [String] {
        var labels: [String] = []
        if asset.mediaSubtypes.contains(.photoScreenshot) {
            labels.append("screenshot")
            labels.append("screen")
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

    private func requestThumbnail(asset: PHAsset, plan: MediaQueryPlan) async -> UIImage? {
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
            options.deliveryMode = plan.needsFineVisualSearch ? .highQualityFormat : .fastFormat
            options.resizeMode = .fast
            options.isNetworkAccessAllowed = false

            let size = plan.needsFineVisualSearch ? CGSize(width: 1024, height: 1024) : CGSize(width: 768, height: 768)
            PHImageManager.default().requestImage(
                for: asset,
                targetSize: size,
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

    private func analyzeImage(_ cgImage: CGImage, plan: MediaQueryPlan) async -> AssetVisionResult {
        let task = Task.detached(priority: .utility) {
            var labels: [String] = []
            var animals: [AnimalDetection] = []
            var humans: [DetectionEvidence] = []
            var faces: [DetectionEvidence] = []
            var textLines: [String] = []
            var requests: [VNRequest] = []

            let animalRequest = plan.needsAnimalDetection ? VNRecognizeAnimalsRequest() : nil
            let humanRequest = plan.needsHumanDetection ? VNDetectHumanRectanglesRequest() : nil
            let faceRequest = plan.needsHumanDetection ? VNDetectFaceRectanglesRequest() : nil
            let textRequest = plan.needsTextRecognition ? VNRecognizeTextRequest() : nil
            let classifyRequest = plan.needsClassification ? VNClassifyImageRequest() : nil

            if let animalRequest {
                requests.append(animalRequest)
            }
            if let humanRequest {
                requests.append(humanRequest)
            }
            if let faceRequest {
                requests.append(faceRequest)
            }
            if let textRequest {
                textRequest.recognitionLevel = .fast
                textRequest.usesLanguageCorrection = false
                requests.append(textRequest)
            }
            if let classifyRequest {
                requests.append(classifyRequest)
            }
            guard !requests.isEmpty else {
                return AssetVisionResult(labels: [], animals: [], humans: [], faces: [], textLines: [])
            }

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform(requests)
            } catch {
                return AssetVisionResult(labels: labels, animals: animals, humans: humans, faces: faces, textLines: textLines)
            }

            for observation in animalRequest?.results ?? [] {
                let area = Double(observation.boundingBox.width * observation.boundingBox.height)
                for label in observation.labels {
                    let normalized = label.identifier.lowercased()
                    animals.append(AnimalDetection(label: normalized, confidence: label.confidence, area: area))
                    labels.append(normalized)
                }
            }

            for observation in humanRequest?.results ?? [] {
                let area = Double(observation.boundingBox.width * observation.boundingBox.height)
                humans.append(DetectionEvidence(confidence: observation.confidence, area: area))
                labels.append("human")
                labels.append("person")
            }

            for observation in faceRequest?.results ?? [] {
                let area = Double(observation.boundingBox.width * observation.boundingBox.height)
                faces.append(DetectionEvidence(confidence: observation.confidence, area: area))
                labels.append("face")
                labels.append("person")
            }

            for observation in textRequest?.results ?? [] {
                if let text = observation.topCandidates(1).first, text.confidence >= 0.25 {
                    textLines.append(text.string.lowercased())
                }
            }

            labels.append(
                contentsOf: (classifyRequest?.results ?? [])
                    .filter { $0.confidence >= 0.35 }
                    .prefix(16)
                    .map { $0.identifier.lowercased() }
            )

            return AssetVisionResult(
                labels: uniqueExpandedLabels(labels),
                animals: animals,
                humans: humans,
                faces: faces,
                textLines: textLines
            )
        }
        return await task.value
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

    private func humanEvidenceScore(_ vision: AssetVisionResult) -> Double {
        let humanScore = vision.humans.map { Double($0.confidence) * 8.0 + min($0.area * 8.0, 2.0) }.max() ?? 0
        let faceScore = vision.faces.map { Double($0.confidence) * 8.0 + min($0.area * 8.0, 2.0) }.max() ?? 0
        return max(humanScore, faceScore)
    }

    private func gameEvidenceScore(_ vision: AssetVisionResult) -> Double {
        let gameAliases = [
            "game",
            "video game",
            "computer game",
            "arcade game",
            "gameplay",
            "gaming"
        ]
        var score = semanticLabelScore(aliases: gameAliases, labels: vision.labels)
        let gameTextCues = [
            "game", "level", "lv", "hp", "score", "start", "battle", "quest", "rank",
            "victory", "defeat", "金币", "钻石", "等级", "战斗", "胜利", "失败", "任务",
            "背包", "开始", "暂停", "关卡", "回合", "角色", "装备", "技能", "血量",
            "分数", "排名", "礼包", "抽卡"
        ]
        score += textScore(aliases: gameTextCues, textLines: vision.textLines)
        return score
    }

    private func semanticLabelScore(aliases: [String], labels: [String]) -> Double {
        guard !aliases.isEmpty else { return 0 }
        var score = 0.0
        for alias in aliases where containsLabel(alias, in: labels) {
            score += alias.count <= 3 ? 4.0 : 3.0
        }
        return score
    }

    private func textScore(aliases: [String], textLines: [String]) -> Double {
        guard !aliases.isEmpty else { return 0 }
        let mergedText = textLines.joined(separator: " ").lowercased()
        guard !mergedText.isEmpty else { return 0 }

        var score = 0.0
        for alias in aliases where mergedText.contains(alias.lowercased()) {
            score += 2.0
        }
        return min(score, 8.0)
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
}

struct MediaSearchOutcome {
    let candidates: [ResourceCandidate]
    let semanticMetrics: SemanticSearchMetrics?
}

private func uniqueExpandedLabels(_ labels: [String]) -> [String] {
    var seen = Set<String>()
    var result: [String] = []
    for label in labels.flatMap(expandVisionIdentifier) {
        if seen.insert(label).inserted {
            result.append(label)
        }
    }
    return result
}

private func expandVisionIdentifier(_ identifier: String) -> [String] {
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

private struct MediaQueryPlan {
    let rawText: String
    let hasSearchTerm: Bool
    let wantsScreenshot: Bool
    let wantsGame: Bool
    let wantsHuman: Bool
    let wantsRecent: Bool
    let wantsToday: Bool
    let animalQuery: AnimalQuery?
    let semanticAliases: [String]

    init(slots: NormalizedSlots) {
        let text = [
            slots.searchKeyword ?? "",
            slots.resourcePhrase,
            slots.resourceType
        ].joined(separator: " ")
        let lowerText = text.lowercased()
        rawText = text
        hasSearchTerm = !(slots.searchKeyword?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true) ||
            !slots.resourcePhrase.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        wantsScreenshot = ["截图", "截屏"].contains(where: { text.contains($0) }) ||
            ["screen", "screenshot"].contains(where: { lowerText.contains($0) })
        wantsGame = ["游戏", "手游"].contains(where: { text.contains($0) }) ||
            ["game", "gameplay"].contains(where: { lowerText.contains($0) })
        wantsHuman = ["人", "人物", "人像", "自拍", "合影"].contains(where: { text.contains($0) }) ||
            ["person", "people", "face"].contains(where: { lowerText.contains($0) })
        wantsRecent = slots.qualifiers.selectionHint.contains("recent")
        wantsToday = slots.qualifiers.time.contains("今天")
        animalQuery = Self.animalQuery(for: text)
        semanticAliases = Self.semanticAliases(for: text, keyword: slots.searchKeyword)
    }

    var needsVision: Bool {
        needsAnimalDetection || needsHumanDetection || needsTextRecognition || needsClassification
    }

    var needsFineVisualSearch: Bool {
        needsAnimalDetection || needsHumanDetection
    }

    var needsAnimalDetection: Bool {
        animalQuery != nil
    }

    var needsHumanDetection: Bool {
        wantsHuman
    }

    var needsTextRecognition: Bool {
        wantsScreenshot || wantsGame || semanticAliases.contains { $0.contains("文字") || $0.contains("text") }
    }

    var needsClassification: Bool {
        wantsGame || (!semanticAliases.isEmpty && animalQuery == nil && !wantsHuman)
    }

    private static func animalQuery(for text: String) -> AnimalQuery? {
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

    private static func semanticAliases(for text: String, keyword: String?) -> [String] {
        var aliases: [String] = []
        if text.contains("游戏") || text.lowercased().contains("game") {
            aliases.append(contentsOf: ["game", "video game", "computer game", "arcade game", "gameplay", "gaming"])
        }
        if text.contains("截图") || text.contains("截屏") {
            aliases.append(contentsOf: ["screenshot", "screen"])
        }
        if text.contains("风景") {
            aliases.append(contentsOf: ["landscape", "sky", "mountain", "beach", "outdoor"])
        }
        if text.contains("会议") {
            aliases.append(contentsOf: ["meeting", "conference", "presentation"])
        }
        if let keyword, !keyword.isEmpty {
            aliases.append(keyword.lowercased())
        }
        return Array(Set(aliases))
    }
}

private struct AssetVisionResult {
    let labels: [String]
    let animals: [AnimalDetection]
    let humans: [DetectionEvidence]
    let faces: [DetectionEvidence]
    let textLines: [String]
}

private struct AnimalDetection {
    let label: String
    let confidence: Float
    let area: Double
}

private struct DetectionEvidence {
    let confidence: Float
    let area: Double
}

private struct AnimalQuery {
    let aliases: [String]
    let minimumConfidence: Float
}
