import Foundation

struct Qualifiers: Codable, Equatable {
    var time: [String]
    var format: [String]
    var selectionHint: [String]

    enum CodingKeys: String, CodingKey {
        case time
        case format
        case selectionHint = "selection_hint"
    }
}

struct NormalizedSlots: Codable, Equatable {
    var resourceType: String
    var resourcePhrase: String
    var searchKeyword: String?
    var target: String
    var targetKeyword: String
    var qualifiers: Qualifiers

    enum CodingKeys: String, CodingKey {
        case resourceType = "resource_type"
        case resourcePhrase = "resource_phrase"
        case searchKeyword = "search_keyword"
        case target
        case targetKeyword = "target_keyword"
        case qualifiers
    }
}

enum SlotNormalizer {
    private static let resourceTypes = [
        "photo": "图片/照片",
        "video": "视频",
        "file": "文件",
        "folder": "文件夹/目录",
        "contact": "联系人/联系方式"
    ]

    private static let resourceTerms: [String: [String]] = [
        "photo": ["GIF动图", "风景照", "图片", "照片", "相片", "截图", "截屏", "自拍", "动图", "相册", "图"],
        "video": ["MOV文件", "MP4文件", "MOV视频", "MP4视频", "短视频", "视频", "录像", "录屏", "短片", "影片", "MP4", "MOV"],
        "file": ["Excel表格", "Word文件", "PDF文件", "PDF文档", "Word文档", "PPT文件", "压缩包", "文件", "文档", "表格", "合同", "报告", "PPT", "PDF", "Word", "Excel"],
        "folder": ["文件夹", "资料夹", "目录"],
        "contact": ["联系人信息", "联系方式", "电子名片", "VCF名片", "vCard联系人", "联系人", "通讯录", "手机号", "电话号码", "微信号", "邮箱地址", "办公地址", "二维码", "电话", "地址", "名片"]
    ]

    private static let genericPrefixes = [
        "选中的", "这个", "那个", "这张", "那张", "这份", "那份", "这段", "那段",
        "这些", "那些", "部分", "多个", "几个", "全部", "所有", "当前"
    ]

    private static let timeTerms = ["今天", "昨天", "前天", "上周", "上个月", "最近", "刚才", "最新", "近期", "去年", "今年", "明天"]
    private static let formatTerms = ["PDF", "Word", "Excel", "PPT", "PNG", "JPG", "GIF", "MP4", "MOV", "VCF", "vCard"]

    static func normalize(intent: String, rawSlots: [String: String]) -> NormalizedSlots? {
        guard intent != "unknown",
              let resourcePhrase = rawSlots["share_content"],
              let target = rawSlots["share_target"] else {
            return nil
        }

        return NormalizedSlots(
            resourceType: resourceTypes[intent] ?? intent,
            resourcePhrase: resourcePhrase,
            searchKeyword: normalizeSearchKeyword(intent: intent, resourcePhrase: resourcePhrase),
            target: target,
            targetKeyword: target,
            qualifiers: extractQualifiers(resourcePhrase)
        )
    }

    private static func normalizeSearchKeyword(intent: String, resourcePhrase: String) -> String? {
        guard intent != "unknown" else { return nil }

        var keyword = resourcePhrase
        for term in (resourceTerms[intent] ?? []).sorted(by: { $0.count > $1.count }) {
            keyword = keyword.replacingOccurrences(of: term, with: "")
        }

        var changed = true
        while changed {
            changed = false
            for prefix in genericPrefixes where keyword.hasPrefix(prefix) {
                keyword.removeFirst(prefix.count)
                changed = true
            }
            for suffix in ["一下", "过去", "过来", "哈"] where keyword.hasSuffix(suffix) {
                keyword.removeLast(suffix.count)
                changed = true
            }
        }

        keyword = keyword
            .replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: " 的。、，,.;；：:"))

        return keyword.isEmpty || ["的", "我", "我的", "这", "那"].contains(keyword) ? nil : keyword
    }

    private static func extractQualifiers(_ text: String) -> Qualifiers {
        var selectionHints: [String] = []
        if ["这张", "这个", "这份", "这段", "这几张", "选中的", "当前", "刚发现的"].contains(where: text.contains) {
            selectionHints.append("current_selection")
        }
        if ["最近", "刚才", "最新", "新拍", "刚保存", "下载好", "下载的", "上次", "今天新增"].contains(where: text.contains) {
            selectionHints.append("recent")
        }
        if ["全部", "所有", "多个", "部分", "十个", "五个", "三张"].contains(where: text.contains) {
            selectionHints.append("all")
        }

        return Qualifiers(
            time: timeTerms.filter(text.contains),
            format: formatTerms.filter(text.contains),
            selectionHint: selectionHints
        )
    }
}
