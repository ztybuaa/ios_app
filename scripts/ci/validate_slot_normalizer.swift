import Foundation

struct NormalizationCase {
    let intent: String
    let phrase: String
    let keyword: String?
    let times: [String]
    let formats: [String]
    let selectionHints: [String]
}

let cases = [
    NormalizationCase(intent: "photo", phrase: "小猫图片", keyword: "小猫", times: [], formats: [], selectionHints: []),
    NormalizationCase(intent: "photo", phrase: "截图", keyword: "截图", times: [], formats: [], selectionHints: []),
    NormalizationCase(intent: "photo", phrase: "拍立得照片", keyword: "拍立得", times: [], formats: [], selectionHints: []),
    NormalizationCase(intent: "photo", phrase: "Tripadvisor照片", keyword: "Tripadvisor", times: [], formats: [], selectionHints: []),
    NormalizationCase(intent: "video", phrase: "最新的视频", keyword: nil, times: ["最新"], formats: [], selectionHints: ["recent"]),
    NormalizationCase(intent: "video", phrase: "录屏视频", keyword: "录屏", times: [], formats: [], selectionHints: []),
    NormalizationCase(intent: "video", phrase: "mp4视频", keyword: nil, times: [], formats: ["MP4"], selectionHints: []),
    NormalizationCase(intent: "file", phrase: "PDF报告", keyword: "报告", times: [], formats: ["PDF"], selectionHints: []),
    NormalizationCase(intent: "file", phrase: "合同文档", keyword: "合同", times: [], formats: [], selectionHints: []),
    NormalizationCase(intent: "folder", phrase: "上周的文件夹", keyword: nil, times: ["上周"], formats: [], selectionHints: []),
    NormalizationCase(intent: "contact", phrase: "今天新增的联系人", keyword: nil, times: ["今天"], formats: [], selectionHints: ["recent"]),
    NormalizationCase(intent: "contact", phrase: "小明的手机号", keyword: "小明", times: [], formats: [], selectionHints: [])
]

for item in cases {
    guard let normalized = SlotNormalizer.normalize(
        intent: item.intent,
        rawSlots: ["share_content": item.phrase, "share_target": "测试目标"]
    ) else {
        fatalError("\(item.intent)/\(item.phrase): normalization unexpectedly returned nil")
    }
    guard normalized.searchKeyword == item.keyword else {
        fatalError("\(item.intent)/\(item.phrase): keyword \(String(describing: normalized.searchKeyword)) != \(String(describing: item.keyword))")
    }
    guard normalized.qualifiers.time == item.times else {
        fatalError("\(item.intent)/\(item.phrase): time \(normalized.qualifiers.time) != \(item.times)")
    }
    guard normalized.qualifiers.format == item.formats else {
        fatalError("\(item.intent)/\(item.phrase): format \(normalized.qualifiers.format) != \(item.formats)")
    }
    guard normalized.qualifiers.selectionHint == item.selectionHints else {
        fatalError(
            "\(item.intent)/\(item.phrase): selection hints \(normalized.qualifiers.selectionHint) != \(item.selectionHints)"
        )
    }
}

print("Swift resource query normalization validation passed (\(cases.count) cases)")
