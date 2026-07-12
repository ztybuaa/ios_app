import Contacts
import Foundation
import Photos

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
                memoryMB: memoryBefore,
                semanticMetrics: nil
            )
        }

        guard let slots else {
            return ResourceSearchResult(
                moduleName: intent,
                statusMessage: "模型未输出完整槽位，无法检索资源候选。",
                resourceCandidates: [],
                targetCandidates: [],
                searchTimeMs: elapsedMs(since: start),
                memoryMB: PerformanceMonitor.currentResidentMemoryMB(),
                semanticMetrics: nil
            )
        }

        switch await permissionGate(intent: intent, slots: slots) {
        case .proceed:
            break
        case .pause(let message):
            return ResourceSearchResult(
                moduleName: intent,
                statusMessage: message,
                resourceCandidates: [],
                targetCandidates: [],
                searchTimeMs: elapsedMs(since: start),
                memoryMB: PerformanceMonitor.currentResidentMemoryMB(),
                semanticMetrics: nil
            )
        case .blocked(let message):
            return ResourceSearchResult(
                moduleName: intent,
                statusMessage: message,
                resourceCandidates: [],
                targetCandidates: [],
                searchTimeMs: elapsedMs(since: start),
                memoryMB: PerformanceMonitor.currentResidentMemoryMB(),
                semanticMetrics: nil
            )
        }

        do {
            let resourceCandidates: [ResourceCandidate]
            let moduleName: String
            let semanticMetrics: SemanticSearchMetrics?

            switch intent {
            case "photo":
                moduleName = "photo"
                let outcome = try await media.search(kind: .photo, slots: slots)
                resourceCandidates = outcome.candidates
                semanticMetrics = outcome.semanticMetrics
            case "video":
                moduleName = "video"
                let outcome = try await media.search(kind: .video, slots: slots)
                resourceCandidates = outcome.candidates
                semanticMetrics = outcome.semanticMetrics
            case "file":
                moduleName = "file"
                resourceCandidates = files.search(kind: .file, slots: slots)
                semanticMetrics = nil
            case "folder":
                moduleName = "folder"
                resourceCandidates = files.search(kind: .folder, slots: slots)
                semanticMetrics = nil
            case "contact":
                moduleName = "contact"
                let query = slots.searchKeyword ?? slots.resourcePhrase
                resourceCandidates = try await contacts.searchContacts(keyword: query)
                semanticMetrics = nil
            default:
                moduleName = intent
                resourceCandidates = []
                semanticMetrics = nil
            }

            var statusMessages = [resourceCandidates.isEmpty ? "未找到资源候选。" : "资源候选检索完成，可点开查看详情。"]
            let targetCandidates: [ResourceCandidate]
            do {
                targetCandidates = try await contacts.searchContacts(keyword: slots.targetKeyword)
                statusMessages.append(targetCandidates.isEmpty ? "未找到目标联系人候选。" : "目标联系人检索完成，可点开查看详情。")
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
                memoryMB: PerformanceMonitor.currentResidentMemoryMB(),
                semanticMetrics: semanticMetrics
            )
        } catch {
            return ResourceSearchResult(
                moduleName: intent,
                statusMessage: error.localizedDescription,
                resourceCandidates: [],
                targetCandidates: [],
                searchTimeMs: elapsedMs(since: start),
                memoryMB: PerformanceMonitor.currentResidentMemoryMB(),
                semanticMetrics: nil
            )
        }
    }

    private func permissionGate(intent: String, slots: NormalizedSlots) async -> PermissionGateResult {
        if intent == "photo" || intent == "video" {
            switch PHPhotoLibrary.authorizationStatus(for: .readWrite) {
            case .authorized, .limited:
                break
            case .notDetermined:
                let status = await PHPhotoLibrary.requestAuthorization(for: .readWrite)
                if status == .authorized || status == .limited {
                    return .pause("相册权限已授权。为避免 iOS 首次授权回调后立刻扫描导致闪退，请再次点击“分析并检索候选”开始真实相册检索。")
                }
                return .blocked("相册权限被拒绝，无法检索照片或视频候选。")
            case .denied, .restricted:
                return .blocked("相册权限不可用，请在系统设置中允许访问照片。")
            @unknown default:
                return .blocked("相册权限状态未知，无法继续检索。")
            }
        }

        let targetKeyword = slots.targetKeyword.trimmingCharacters(in: .whitespacesAndNewlines)
        if intent == "contact" || !targetKeyword.isEmpty {
            switch CNContactStore.authorizationStatus(for: .contacts) {
            case .authorized, .limited:
                break
            case .notDetermined:
                do {
                    let granted = try await CNContactStore().requestAccess(for: .contacts)
                    if granted {
                        return .pause("通讯录权限已授权。为避免 iOS 首次授权回调后立刻扫描导致闪退，请再次点击“分析并检索候选”开始真实联系人检索。")
                    }
                    return .blocked("通讯录权限被拒绝，无法检索联系人候选。")
                } catch {
                    return .blocked("通讯录授权失败：\(error.localizedDescription)")
                }
            case .denied, .restricted:
                return .blocked("通讯录权限不可用，请在系统设置中允许访问联系人。")
            @unknown default:
                return .blocked("通讯录权限状态未知，无法继续检索。")
            }
        }

        return .proceed
    }

    private func elapsedMs(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1000
    }
}

private enum PermissionGateResult {
    case proceed
    case pause(String)
    case blocked(String)
}
