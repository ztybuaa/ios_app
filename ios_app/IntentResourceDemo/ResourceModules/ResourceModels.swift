import Foundation

enum CandidateKind: String, Codable {
    case photo
    case video
    case file
    case folder
    case contact
}

struct ResourceCandidate: Identifiable, Codable {
    let id: String
    let kind: CandidateKind
    let title: String
    let subtitle: String
    let detail: String
    let score: Double
    let debugInfo: String
}

struct ResourceSearchResult {
    let moduleName: String
    let statusMessage: String
    let resourceCandidates: [ResourceCandidate]
    let targetCandidates: [ResourceCandidate]
    let searchTimeMs: Double
    let memoryMB: Double?
    let semanticMetrics: SemanticSearchMetrics?
}

struct SemanticSearchMetrics {
    let modelLoadTimeMs: Double
    let totalWallMs: Double
    let fullPassWallMs: Double
    let rerankWallMs: Double
    let scannedAssetCount: Int
    let shortlistAssetCount: Int
    let memoryCacheHits: Int
    let diskCacheHits: Int
    let cacheMisses: Int
    let corruptCacheEvictions: Int
    let cacheStorageUnavailable: Int
    let imageRequestFailures: Int
    let coarsePredictionCount: Int
    let qualityPredictionCount: Int
}

struct IndexedResourceItem: Codable, Identifiable {
    let id: String
    let kind: CandidateKind
    let title: String
    let path: String
    let summary: String
    let tags: [String]
    let format: String?
    let updatedAt: String?
}

struct SampleResourceIndex: Codable {
    let files: [IndexedResourceItem]
    let folders: [IndexedResourceItem]
}

enum CandidateScorer {
    static func score(
        keyword: String?,
        phrase: String,
        targetText: String,
        tags: [String] = [],
        qualifiers: Qualifiers? = nil
    ) -> Double {
        let queryTerms = normalizedTerms(keyword: keyword, phrase: phrase)
        let haystack = ([targetText] + tags).joined(separator: " ").lowercased()
        var score = 0.0

        for term in queryTerms {
            if haystack.contains(term.lowercased()) {
                score += term.count <= 1 ? 0.5 : 2.0
            }
        }

        if let qualifiers {
            for format in qualifiers.format where haystack.contains(format.lowercased()) {
                score += 1.0
            }
            if qualifiers.selectionHint.contains("recent") {
                score += 0.25
            }
        }

        return score
    }

    static func normalizedTerms(keyword: String?, phrase: String) -> [String] {
        let raw = [keyword, phrase].compactMap { $0 }
        var terms: [String] = []
        for item in raw {
            let pieces = item
                .replacingOccurrences(of: "/", with: " ")
                .replacingOccurrences(of: "_", with: " ")
                .split(separator: " ")
                .map(String.init)
            terms.append(contentsOf: pieces.isEmpty ? [item] : pieces)
        }
        return Array(Set(terms.filter { !$0.isEmpty }))
    }
}
