import Foundation

func slots(
    phrase: String,
    keyword: String?,
    formats: [String] = [],
    times: [String] = [],
    selectionHints: [String] = []
) -> NormalizedSlots {
    NormalizedSlots(
        resourceType: "测试资源",
        resourcePhrase: phrase,
        searchKeyword: keyword,
        target: "测试目标",
        targetKeyword: "测试目标",
        qualifiers: Qualifiers(time: times, format: formats, selectionHint: selectionHints)
    )
}

func item(
    id: String,
    kind: CandidateKind,
    title: String,
    summary: String,
    tags: [String],
    format: String? = nil,
    updatedAt: String? = "2026-07-12",
    contentTerms: [String]? = nil,
    childTerms: [String]? = nil
) -> IndexedResourceItem {
    IndexedResourceItem(
        id: id,
        kind: kind,
        title: title,
        path: "测试/\(title)",
        summary: summary,
        tags: tags,
        format: format,
        updatedAt: updatedAt,
        contentTerms: contentTerms,
        childTerms: childTerms
    )
}

let contract = item(
    id: "contract",
    kind: .file,
    title: "华东供应商合同.pdf",
    summary: "供应协议和法务修订记录",
    tags: ["合同", "供应商", "华东"],
    format: "PDF",
    contentTerms: ["付款条款", "交付周期"]
)
let unrelated = item(
    id: "budget",
    kind: .file,
    title: "年度预算.xlsx",
    summary: "部门预算和支出",
    tags: ["预算", "财务"],
    format: "Excel"
)
let contractQuery = slots(phrase: "华东供应合同PDF文件", keyword: "华东供应合同", formats: ["PDF"])
guard let contractScore = CandidateScorer.score(indexedItem: contract, slots: contractQuery),
      CandidateScorer.score(indexedItem: unrelated, slots: contractQuery) == nil,
      contractScore.value >= 3 else {
    fatalError("field-aware file scoring failed")
}

let folder = item(
    id: "folder",
    kind: .folder,
    title: "供应商合同归档",
    summary: "供应商协议归档",
    tags: ["供应商", "合同"],
    childTerms: ["付款条款", "盖章扫描件"]
)
let childQuery = slots(phrase: "有付款条款的资料夹", keyword: "付款条款")
guard let childScore = CandidateScorer.score(indexedItem: folder, slots: childQuery),
      childScore.debugInfo.contains("child_terms") else {
    fatalError("folder child evidence was not indexed")
}

let unknownQuery = slots(phrase: "量子芯片实验记录文件", keyword: "量子芯片实验记录")
guard CandidateScorer.score(indexedItem: contract, slots: unknownQuery) == nil,
      CandidateScorer.score(indexedItem: unrelated, slots: unknownQuery) == nil else {
    fatalError("open-set query should have been rejected")
}

print("Swift candidate scorer validation passed")
