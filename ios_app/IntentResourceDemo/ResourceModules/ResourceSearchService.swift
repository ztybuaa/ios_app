import Foundation

final class ResourceSearchService {
    private let contacts = ContactResourceModule()
    private let media = MediaResourceModule()
    private let files = FileFolderResourceModule()

    func search(intent: String, slots: NormalizedSlots?) async -> ResourceSearchResult {
        let start = CFAbsoluteTimeGetCurrent()
        let memoryBefore = PerformanceMonitor.currentResidentMemoryMB()

        guard intent != "unknown" else {
            return ResourceSearchResult(
                moduleName: "unknown",
                statusMessage: "unknown 输入已拦截，不进入资源检索流程。",
                resourceCandidates: [],
                targetCandidates: [],
                searchTimeMs: 0,
                memoryMB: memoryBefore
            )
        }

        guard let slots else {
            return ResourceSearchResult(
                moduleName: intent,
                statusMessage: "模型未输出完整槽位，无法检索资源候选。",
                resourceCandidates: [],
                targetCandidates: [],
                searchTimeMs: elapsedMs(since: start),
                memoryMB: PerformanceMonitor.currentResidentMemoryMB()
            )
        }

        do {
            let resourceCandidates: [ResourceCandidate]
            let moduleName: String

            switch intent {
            case "photo":
                moduleName = "photo"
                resourceCandidates = try await media.search(kind: .photo, slots: slots)
            case "video":
                moduleName = "video"
                resourceCandidates = try await media.search(kind: .video, slots: slots)
            case "file":
                moduleName = "file"
                resourceCandidates = files.search(kind: .file, slots: slots)
            case "folder":
                moduleName = "folder"
                resourceCandidates = files.search(kind: .folder, slots: slots)
            case "contact":
                moduleName = "contact"
                let query = slots.searchKeyword ?? slots.resourcePhrase
                resourceCandidates = try await contacts.searchContacts(keyword: query)
            default:
                moduleName = intent
                resourceCandidates = []
            }

            var statusMessages = [resourceCandidates.isEmpty ? "未找到资源候选。" : "资源候选检索完成。"]
            let targetCandidates: [ResourceCandidate]
            do {
                targetCandidates = try await contacts.searchContacts(keyword: slots.targetKeyword)
                statusMessages.append(targetCandidates.isEmpty ? "未找到目标联系人候选。" : "目标联系人检索完成。")
            } catch {
                targetCandidates = []
                statusMessages.append("目标联系人检索失败：\(error.localizedDescription)")
            }

            return ResourceSearchResult(
                moduleName: moduleName,
                statusMessage: statusMessages.joined(separator: " "),
                resourceCandidates: resourceCandidates,
                targetCandidates: targetCandidates,
                searchTimeMs: elapsedMs(since: start),
                memoryMB: PerformanceMonitor.currentResidentMemoryMB()
            )
        } catch {
            return ResourceSearchResult(
                moduleName: intent,
                statusMessage: error.localizedDescription,
                resourceCandidates: [],
                targetCandidates: [],
                searchTimeMs: elapsedMs(since: start),
                memoryMB: PerformanceMonitor.currentResidentMemoryMB()
            )
        }
    }

    private func elapsedMs(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1000
    }
}
