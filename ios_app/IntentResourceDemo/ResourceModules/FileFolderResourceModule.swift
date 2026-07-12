import Foundation

final class FileFolderResourceModule {
    private let index: Result<SampleResourceIndex, Error>

    init() {
        do {
            index = .success(try Self.loadIndex())
        } catch {
            index = .failure(error)
        }
    }

    func search(kind: CandidateKind, slots: NormalizedSlots, limit: Int = 12) throws -> [ResourceCandidate] {
        guard kind == .file || kind == .folder else {
            throw DemoError.invalidInput
        }

        let index = try index.get()
        if kind == .folder, !CandidateScorer.requestedFormats(slots: slots).isEmpty {
            throw DemoError.resourceUnavailable(
                "当前文件夹索引没有子文件格式信息，无法可靠执行格式筛选。"
            )
        }

        let source: [IndexedResourceItem] = kind == .folder ? index.folders : index.files
        let candidates = source.compactMap { item -> (ResourceCandidate, IndexedResourceScore)? in
            guard let match = CandidateScorer.score(indexedItem: item, slots: slots) else {
                return nil
            }

            let candidate = ResourceCandidate(
                id: item.id,
                kind: kind,
                title: item.title,
                subtitle: item.path,
                detail: item.summary,
                score: match.value,
                debugInfo: match.debugInfo
            )
            return (candidate, match)
        }

        return candidates
            .sorted { lhs, rhs in
                if lhs.1.prioritizesRecency,
                   lhs.1.updatedAt != rhs.1.updatedAt {
                    return (lhs.1.updatedAt ?? .distantPast) > (rhs.1.updatedAt ?? .distantPast)
                }
                if lhs.0.score != rhs.0.score {
                    return lhs.0.score > rhs.0.score
                }
                return lhs.0.title.localizedStandardCompare(rhs.0.title) == .orderedAscending
            }
            .prefix(limit)
            .map { $0.0 }
    }

    private static func loadIndex() throws -> SampleResourceIndex {
        guard let url = Bundle.main.url(forResource: "sample_resource_index", withExtension: "json") else {
            throw DemoError.resourceUnavailable("样例文件索引未加入 Bundle。")
        }
        let data = try Data(contentsOf: url)
        let index = try JSONDecoder().decode(SampleResourceIndex.self, from: data)
        try validate(index)
        return index
    }

    private static func validate(_ index: SampleResourceIndex) throws {
        let items = index.files + index.folders
        let ids = items.map(\.id)
        guard ids.count == Set(ids).count else {
            throw DemoError.resourceUnavailable("样例文件索引包含重复标识。")
        }
        guard index.files.allSatisfy({ $0.kind == .file }),
              index.folders.allSatisfy({ $0.kind == .folder }) else {
            throw DemoError.resourceUnavailable("样例文件索引的资源类型与分组不一致。")
        }
        guard items.allSatisfy({ item in
            guard let updatedAt = item.updatedAt else { return true }
            return CandidateScorer.parseIndexedDate(updatedAt) != nil
        }) else {
            throw DemoError.resourceUnavailable("样例文件索引包含无效的 updatedAt 日期。")
        }
    }
}
