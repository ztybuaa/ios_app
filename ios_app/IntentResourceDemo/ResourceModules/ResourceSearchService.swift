import Foundation

final class ResourceSearchService {
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

        let moduleName = intent
        let resourceCandidates = DemoResourceCatalog.searchResource(intent: intent, slots: slots)
        let targetCandidates = DemoResourceCatalog.searchTargets(keyword: slots.targetKeyword)
        let statusMessages = [
            "稳定诊断模式：本次不访问系统相册或通讯录，只检索内置候选。",
            resourceCandidates.isEmpty ? "未找到资源候选。" : "资源候选检索完成。",
            targetCandidates.isEmpty ? "未找到目标联系人候选。" : "目标联系人检索完成。"
        ]

        return ResourceSearchResult(
            moduleName: moduleName,
            statusMessage: statusMessages.joined(separator: " "),
            resourceCandidates: resourceCandidates,
            targetCandidates: targetCandidates,
            searchTimeMs: elapsedMs(since: start),
            memoryMB: PerformanceMonitor.currentResidentMemoryMB()
        )
    }

    private func elapsedMs(since start: CFAbsoluteTime) -> Double {
        (CFAbsoluteTimeGetCurrent() - start) * 1000
    }
}

private enum DemoResourceCatalog {
    private struct Item {
        let id: String
        let kind: CandidateKind
        let title: String
        let subtitle: String
        let detail: String
        let tags: [String]
    }

    static func searchResource(intent: String, slots: NormalizedSlots, limit: Int = 12) -> [ResourceCandidate] {
        guard let kind = kind(for: intent) else { return [] }
        let keyword = intent == "contact" ? (slots.searchKeyword ?? slots.resourcePhrase) : slots.searchKeyword
        return search(
            kind: kind,
            keyword: keyword,
            phrase: slots.resourcePhrase,
            qualifiers: slots.qualifiers,
            limit: limit
        )
    }

    static func searchTargets(keyword: String, limit: Int = 8) -> [ResourceCandidate] {
        search(
            kind: .contact,
            keyword: keyword,
            phrase: keyword,
            qualifiers: nil,
            limit: limit
        )
    }

    private static func search(
        kind: CandidateKind,
        keyword: String?,
        phrase: String,
        qualifiers: Qualifiers?,
        limit: Int
    ) -> [ResourceCandidate] {
        candidates
            .filter { $0.kind == kind }
            .map { item in
                let score = CandidateScorer.score(
                    keyword: keyword,
                    phrase: phrase,
                    targetText: "\(item.title) \(item.subtitle) \(item.detail)",
                    tags: item.tags,
                    qualifiers: qualifiers
                )
                return ResourceCandidate(
                    id: item.id,
                    kind: item.kind,
                    title: item.title,
                    subtitle: item.subtitle,
                    detail: item.detail,
                    score: score,
                    debugInfo: "matched bundled demo catalog; no Photos/Contacts API access"
                )
            }
            .filter { $0.score > 0 || keyword == nil }
            .sorted { $0.score == $1.score ? $0.title < $1.title : $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }

    private static func kind(for intent: String) -> CandidateKind? {
        switch intent {
        case "photo": return .photo
        case "video": return .video
        case "file": return .file
        case "folder": return .folder
        case "contact": return .contact
        default: return nil
        }
    }

    private static let candidates: [Item] = [
        Item(
            id: "demo-photo-dog-001",
            kind: .photo,
            title: "小狗公园照片",
            subtitle: "相册/最近项目/今天 14:20",
            detail: "一张小狗在公园草地上的照片，适合测试“小狗图片”检索。",
            tags: ["小狗", "狗", "宠物", "照片", "图片", "最近", "今天"]
        ),
        Item(
            id: "demo-photo-screenshot-001",
            kind: .photo,
            title: "订单截图",
            subtitle: "相册/截图/昨天 20:10",
            detail: "包含订单号、金额和收货信息的手机截图。",
            tags: ["截图", "订单", "图片", "照片", "昨天"]
        ),
        Item(
            id: "demo-photo-travel-001",
            kind: .photo,
            title: "旅行风景照",
            subtitle: "相册/旅行/2026-06-28",
            detail: "海边、天空和城市夜景照片合集中的候选。",
            tags: ["旅行", "风景", "照片", "图片", "海边"]
        ),
        Item(
            id: "demo-video-meeting-001",
            kind: .video,
            title: "项目会议录屏",
            subtitle: "视频/会议/08:32",
            detail: "项目同步会的屏幕录制，包含进度和风险讨论。",
            tags: ["会议", "录屏", "视频", "项目", "MP4"]
        ),
        Item(
            id: "demo-video-dog-001",
            kind: .video,
            title: "小狗短视频",
            subtitle: "视频/宠物/00:18",
            detail: "小狗在客厅玩球的短视频。",
            tags: ["小狗", "狗", "宠物", "短视频", "视频"]
        ),
        Item(
            id: "demo-file-quote-001",
            kind: .file,
            title: "华东客户报价单.pdf",
            subtitle: "On My iPhone/DemoFiles/华东客户报价单.pdf",
            detail: "包含客户报价、折扣、交付周期和付款条款。",
            tags: ["报价单", "客户", "PDF", "销售", "文件"]
        ),
        Item(
            id: "demo-file-contract-001",
            kind: .file,
            title: "供应商合同.docx",
            subtitle: "On My iPhone/DemoFiles/供应商合同.docx",
            detail: "供应商合作合同与法务修订记录。",
            tags: ["合同", "供应商", "Word", "法务", "文件"]
        ),
        Item(
            id: "demo-file-report-001",
            kind: .file,
            title: "项目进度报告.pptx",
            subtitle: "On My iPhone/DemoFiles/项目进度报告.pptx",
            detail: "项目里程碑、风险、预算和下一步计划。",
            tags: ["项目", "报告", "PPT", "进度", "文件"]
        ),
        Item(
            id: "demo-folder-project-001",
            kind: .folder,
            title: "项目资料",
            subtitle: "On My iPhone/DemoFolders/项目资料",
            detail: "项目计划、会议纪要、进度报告和风险清单。",
            tags: ["项目资料", "项目", "资料", "会议纪要", "文件夹"]
        ),
        Item(
            id: "demo-folder-photo-001",
            kind: .folder,
            title: "照片备份",
            subtitle: "On My iPhone/DemoFolders/照片备份",
            detail: "旅行照片、会议照片、截图和精选图片备份。",
            tags: ["照片备份", "照片", "旅行", "截图", "文件夹"]
        ),
        Item(
            id: "demo-contact-xiaoming-001",
            kind: .contact,
            title: "小明",
            subtitle: "138-0000-0001",
            detail: "同学联系人，小明，微信 xiaoming_demo。",
            tags: ["小明", "同学", "联系人", "电话", "微信"]
        ),
        Item(
            id: "demo-contact-xiaohong-001",
            kind: .contact,
            title: "小红",
            subtitle: "139-0000-0002",
            detail: "项目联系人，小红，邮箱 xiaohong@example.com。",
            tags: ["小红", "项目", "联系人", "电话", "邮箱"]
        ),
        Item(
            id: "demo-contact-lisi-001",
            kind: .contact,
            title: "李四",
            subtitle: "lisi@example.com",
            detail: "供应商联系人，负责合同和报价。",
            tags: ["李四", "供应商", "合同", "报价", "联系人"]
        )
    ]
}
