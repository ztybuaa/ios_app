import Foundation

final class FileFolderResourceModule {
    private let index: SampleResourceIndex

    init() {
        self.index = (try? Self.loadIndex()) ?? SampleResourceIndex(files: [], folders: [])
    }

    func search(kind: CandidateKind, slots: NormalizedSlots, limit: Int = 12) -> [ResourceCandidate] {
        let source: [IndexedResourceItem] = kind == .folder ? index.folders : index.files
        let candidates = source.map { item -> ResourceCandidate in
            let score = CandidateScorer.score(
                keyword: slots.searchKeyword,
                phrase: slots.resourcePhrase,
                targetText: "\(item.title) \(item.path) \(item.summary) \(item.format ?? "")",
                tags: item.tags,
                qualifiers: slots.qualifiers
            )

            return ResourceCandidate(
                id: item.id,
                kind: kind,
                title: item.title,
                subtitle: item.path,
                detail: item.summary,
                score: score,
                debugInfo: "matched bundled index title/path/summary/tags"
            )
        }

        return candidates
            .filter { $0.score > 0 || slots.searchKeyword == nil }
            .sorted { $0.score == $1.score ? $0.title < $1.title : $0.score > $1.score }
            .prefix(limit)
            .map { $0 }
    }

    private static func loadIndex() throws -> SampleResourceIndex {
        guard let url = Bundle.main.url(forResource: "sample_resource_index", withExtension: "json") else {
            throw DemoError.resourceUnavailable("样例文件索引未加入 Bundle。")
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(SampleResourceIndex.self, from: data)
    }
}
