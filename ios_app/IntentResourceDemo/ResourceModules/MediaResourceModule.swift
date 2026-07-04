import Foundation
import Photos
import UIKit
import Vision

final class MediaResourceModule {
    func search(kind: CandidateKind, slots: NormalizedSlots, limit: Int = 12) async throws -> [ResourceCandidate] {
        try await ensureAccess()

        let fetchOptions = PHFetchOptions()
        fetchOptions.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: false)]
        fetchOptions.fetchLimit = limit
        fetchOptions.predicate = NSPredicate(
            format: "mediaType == %d",
            kind == .video ? PHAssetMediaType.video.rawValue : PHAssetMediaType.image.rawValue
        )

        let assets = PHAsset.fetchAssets(with: fetchOptions)
        var candidates: [ResourceCandidate] = []

        for index in 0..<assets.count {
            let asset = assets.object(at: index)
            let labels = await labelsForAsset(asset)
            let score = score(asset: asset, labels: labels, slots: slots)
            if score > 0 || slots.searchKeyword == nil || slots.qualifiers.selectionHint.contains("recent") {
                candidates.append(makeCandidate(asset: asset, kind: kind, labels: labels, score: score))
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

    private func score(asset: PHAsset, labels: [String], slots: NormalizedSlots) -> Double {
        let keyword = slots.searchKeyword
        let keywordAliases = aliases(for: keyword)
        let labelText = labels.joined(separator: " ")
        var score = CandidateScorer.score(
            keyword: keyword,
            phrase: slots.resourcePhrase,
            targetText: labelText,
            tags: keywordAliases,
            qualifiers: slots.qualifiers
        )

        if slots.qualifiers.selectionHint.contains("recent") {
            score += 1.0
        }
        if slots.qualifiers.time.contains("今天"), Calendar.current.isDateInToday(asset.creationDate ?? .distantPast) {
            score += 1.0
        }
        if slots.resourcePhrase.contains("截图"), asset.mediaSubtypes.contains(.photoScreenshot) {
            score += 3.0
        }
        if labels.contains(where: keywordAliases.contains) {
            score += 2.0
        }

        return score
    }

    private func makeCandidate(asset: PHAsset, kind: CandidateKind, labels: [String], score: Double) -> ResourceCandidate {
        let date = asset.creationDate.map { DateFormatter.localizedString(from: $0, dateStyle: .short, timeStyle: .short) } ?? "未知时间"
        let dimensions = "\(asset.pixelWidth)x\(asset.pixelHeight)"
        let title = kind == .video ? "视频 \(date)" : "照片 \(date)"

        return ResourceCandidate(
            id: asset.localIdentifier,
            kind: kind,
            title: title,
            subtitle: dimensions,
            detail: labels.isEmpty ? "无视觉标签" : labels.prefix(5).joined(separator: ", "),
            score: score,
            debugInfo: "matched Photos metadata + Vision image labels"
        )
    }

    private func labelsForAsset(_ asset: PHAsset) async -> [String] {
        guard let image = await requestImage(asset: asset), let cgImage = image.cgImage else {
            return []
        }

        return await Task.detached(priority: .utility) {
            let request = VNClassifyImageRequest()
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
                return (request.results ?? [])
                    .prefix(8)
                    .map { $0.identifier.lowercased() }
            } catch {
                return []
            }
        }.value
    }

    private func requestImage(asset: PHAsset) async -> UIImage? {
        await withCheckedContinuation { continuation in
            let resumeLock = NSLock()
            var didResume = false

            func resumeOnce(_ image: UIImage?) {
                resumeLock.lock()
                defer { resumeLock.unlock() }
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
                targetSize: CGSize(width: 224, height: 224),
                contentMode: .aspectFill,
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

    private func aliases(for keyword: String?) -> [String] {
        guard let keyword else { return [] }
        let table: [String: [String]] = [
            "小狗": ["dog", "puppy", "canine"],
            "狗": ["dog", "puppy", "canine"],
            "猫": ["cat", "kitten"],
            "小猫": ["cat", "kitten"],
            "人": ["person", "people", "human"],
            "会议": ["meeting", "conference", "presentation"],
            "旅行": ["travel", "landscape", "outdoor"],
            "风景": ["landscape", "sky", "mountain", "beach"],
            "截图": ["screenshot"]
        ]
        return table[keyword] ?? [keyword.lowercased()]
    }
}
