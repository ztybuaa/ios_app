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
    let contentTerms: [String]?
    let childTerms: [String]?
}

struct SampleResourceIndex: Codable {
    let files: [IndexedResourceItem]
    let folders: [IndexedResourceItem]
}

struct IndexedResourceScore {
    let value: Double
    let lexicalValue: Double
    let updatedAt: Date?
    let prioritizesRecency: Bool
    let debugInfo: String
}

struct CandidateTextField {
    let name: String
    let values: [String]
    let weight: Double
}

struct CandidateLexicalScore {
    let value: Double
    let matchedFields: [String]
}

enum CandidateScorer {
    private static let minimumLexicalScore = 3.0
    private static let minimumFieldSimilarity = 0.45

    private static let resourceTerms = [
        "Word文件", "Excel表格", "PDF文件", "PPT文件", "文件夹", "资料夹",
        "文件", "文档", "表格", "目录"
    ]
    private static let timeTerms = [
        "刚更新", "刚保存", "最近的", "最新的", "上个月", "最近", "最新", "近期",
        "刚才", "今天", "昨天", "前天", "本周", "上周", "本月", "今年", "去年", "明天"
    ]
    private static let commandTerms = [
        "请帮我", "帮我", "查找", "搜索", "打开", "找到", "选择", "发给", "发送", "给我", "找", "把"
    ]
    private static let formatAliases: [String: [String]] = [
        "markdown": ["markdown", "md"],
        "pdf": ["pdf"],
        "word": ["word", "docx", "doc"],
        "excel": ["excel", "xlsx", "xls"],
        "ppt": ["powerpoint", "pptx", "ppt"],
        "text": ["text", "txt"],
        "vcard": ["vcard", "vcf"],
        "jpg": ["jpg", "jpeg"],
        "png": ["png"],
        "gif": ["gif"],
        "mp4": ["mp4"],
        "mov": ["mov"],
        "avi": ["avi"],
        "mkv": ["mkv"],
        "csv": ["csv"],
        "zip": ["zip"],
        "rar": ["rar"],
        "7z": ["7z"],
        "json": ["json"],
        "xml": ["xml"],
        "sql": ["sql"],
        "ipa": ["ipa"],
        "apk": ["apk"],
        "dmg": ["dmg"],
        "exe": ["exe"]
    ]

    static func score(
        indexedItem item: IndexedResourceItem,
        slots: NormalizedSlots,
        referenceDate: Date = Date()
    ) -> IndexedResourceScore? {
        let plan = indexedResourcePlan(kind: item.kind, slots: slots, referenceDate: referenceDate)
        guard matchesStructuredFilters(item: item, plan: plan) else {
            return nil
        }

        let lexicalMatch = plan.subject.isEmpty
            ? CandidateLexicalScore(value: 0, matchedFields: [])
            : lexicalScore(subject: plan.subject, fields: indexedFields(for: item))
        guard let lexicalMatch else { return nil }
        let lexicalValue = lexicalMatch.value

        let updatedAt = item.updatedAt.flatMap(parseIndexedDate)
        var score = lexicalValue
        if plan.appliesFormatFilter {
            score += 2.0
        }
        if plan.dateInterval != nil {
            score += 1.0
        }

        let mode = plan.subject.isEmpty ? "browse" : "lexical"
        let filters = [
            plan.appliesFormatFilter ? "format=\(plan.formats.sorted().joined(separator: ","))" : nil,
            plan.dateInterval == nil ? nil : "date=hard",
            plan.prioritizesRecency ? "sort=recent" : nil
        ].compactMap { $0 }.joined(separator: " ")
        let matchedFieldText = lexicalMatch.matchedFields.isEmpty
            ? ""
            : " fields=\(lexicalMatch.matchedFields.joined(separator: ","))"
        let debugInfo = String(
            format: "field-aware %@ score=%.3f lexical=%.3f%@%@",
            mode,
            score,
            lexicalValue,
            matchedFieldText,
            filters.isEmpty ? "" : " \(filters)"
        )

        return IndexedResourceScore(
            value: score,
            lexicalValue: lexicalValue,
            updatedAt: updatedAt,
            prioritizesRecency: plan.prioritizesRecency,
            debugInfo: debugInfo
        )
    }

    static func lexicalScore(
        subject: String,
        fields: [CandidateTextField]
    ) -> CandidateLexicalScore? {
        let fragments = Array(Set(([subject] + subject.split(whereSeparator: { $0.isWhitespace }).map(String.init))))
        guard fragments.contains(where: { !compactText($0).isEmpty }) else { return nil }

        var matches: [(name: String, value: Double)] = []
        for field in fields {
            var best = 0.0
            for value in field.values {
                let similarity = fragments.map { lexicalSimilarity(query: $0, candidate: value) }.max() ?? 0
                if similarity >= minimumFieldSimilarity {
                    best = max(best, field.weight * similarity)
                }
            }
            if best > 0 {
                matches.append((field.name, best))
            }
        }

        guard !matches.isEmpty else { return nil }
        matches.sort { lhs, rhs in
            lhs.value == rhs.value ? lhs.name < rhs.name : lhs.value > rhs.value
        }
        let value = matches[0].value + 0.75 * matches.dropFirst().reduce(0) { $0 + $1.value }
        guard value >= minimumLexicalScore else { return nil }
        return CandidateLexicalScore(value: value, matchedFields: matches.map { $0.name })
    }

    static func parseIndexedDate(_ value: String) -> Date? {
        indexedDateFormatter.date(from: value)
    }

    static func requestedFormats(slots: NormalizedSlots) -> Set<String> {
        let queryText = [slots.searchKeyword, slots.resourcePhrase]
            .compactMap { $0 }
            .joined(separator: " ")
        return Set(slots.qualifiers.format.map(canonicalFormat).filter { !$0.isEmpty })
            .union(detectedFormats(in: queryText))
    }

    private struct IndexedResourceQueryPlan {
        let subject: String
        let formats: Set<String>
        let appliesFormatFilter: Bool
        let dateInterval: DateInterval?
        let prioritizesRecency: Bool
    }

    private static let indexedDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.calendar = Calendar(identifier: .gregorian)
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .autoupdatingCurrent
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.isLenient = false
        return formatter
    }()

    private static func indexedResourcePlan(
        kind: CandidateKind,
        slots: NormalizedSlots,
        referenceDate: Date
    ) -> IndexedResourceQueryPlan {
        let rawSubject = (slots.searchKeyword ?? slots.resourcePhrase)
            .precomposedStringWithCompatibilityMapping
            .lowercased()
        var subject = rawSubject
        let queryFormats = requestedFormats(slots: slots)
        let appliesFormatFilter = kind == .file && !queryFormats.isEmpty

        let requestedFormatTerms = appliesFormatFilter
            ? queryFormats.flatMap { formatAliases[$0] ?? [$0] }
            : []
        let removalTerms = timeTerms + requestedFormatTerms
        for term in removalTerms.sorted(by: { $0.count > $1.count }) {
            subject = subject.replacingOccurrences(of: term.lowercased(), with: "")
        }
        var removedResourceTerm = true
        while removedResourceTerm {
            removedResourceTerm = false
            for term in resourceTerms.sorted(by: { $0.count > $1.count })
                where subject == term.lowercased() || subject.hasSuffix(term.lowercased()) {
                subject.removeLast(term.count)
                removedResourceTerm = true
                break
            }
        }
        var removedCommand = true
        while removedCommand {
            removedCommand = false
            for command in commandTerms.sorted(by: { $0.count > $1.count }) where subject.hasPrefix(command) {
                subject.removeFirst(command.count)
                removedCommand = true
                break
            }
        }
        subject = subject
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: " 的、，,。.;；：:"))

        let dateInterval = structuredDateInterval(
            terms: slots.qualifiers.time,
            referenceDate: referenceDate
        )
        let prioritizesRecency = slots.qualifiers.selectionHint.contains("recent")
            || slots.qualifiers.time.contains(where: { ["最近", "最新", "近期", "刚才"].contains($0) })

        return IndexedResourceQueryPlan(
            subject: subject,
            formats: queryFormats,
            appliesFormatFilter: appliesFormatFilter,
            dateInterval: dateInterval,
            prioritizesRecency: prioritizesRecency
        )
    }

    private static func matchesStructuredFilters(
        item: IndexedResourceItem,
        plan: IndexedResourceQueryPlan
    ) -> Bool {
        if plan.appliesFormatFilter {
            guard let format = item.format.map(canonicalFormat), plan.formats.contains(format) else {
                return false
            }
        }

        if let interval = plan.dateInterval {
            guard let rawDate = item.updatedAt,
                  let updatedAt = parseIndexedDate(rawDate),
                  updatedAt >= interval.start,
                  updatedAt < interval.end else {
                return false
            }
        }
        return true
    }

    private static func canonicalFormat(_ value: String) -> String {
        let normalized = compactText(value)
        for (canonical, aliases) in formatAliases where aliases.contains(normalized) {
            return canonical
        }
        return normalized
    }

    private static func detectedFormats(in value: String) -> Set<String> {
        let normalized = value.precomposedStringWithCompatibilityMapping.lowercased()
        return Set(formatAliases.compactMap { canonical, aliases in
            let matched = aliases.contains { alias in
                let escaped = NSRegularExpression.escapedPattern(for: alias)
                let pattern = "(?<![a-z0-9])\(escaped)(?![a-z0-9])"
                return normalized.range(of: pattern, options: .regularExpression) != nil
            }
            return matched ? canonical : nil
        })
    }

    private static func indexedFields(for item: IndexedResourceItem) -> [CandidateTextField] {
        var fields = [
            CandidateTextField(name: "title", values: [item.title], weight: 8.0),
            CandidateTextField(name: "tags", values: item.tags, weight: 5.0),
            CandidateTextField(name: "summary", values: [item.summary], weight: 3.5),
            CandidateTextField(name: "path", values: [item.path], weight: 3.5)
        ]
        if item.kind == .file, let contentTerms = item.contentTerms {
            fields.append(CandidateTextField(name: "content_terms", values: contentTerms, weight: 4.0))
        }
        if item.kind == .folder, let childTerms = item.childTerms {
            fields.append(CandidateTextField(name: "child_terms", values: childTerms, weight: 4.5))
        }
        return fields
    }

    private static func lexicalSimilarity(query rawQuery: String, candidate rawCandidate: String) -> Double {
        let query = compactText(rawQuery)
        let candidate = compactText(rawCandidate)
        guard !query.isEmpty, !candidate.isEmpty else { return 0 }
        if query == candidate { return 1 }
        if candidate.contains(query) {
            return min(0.99, 0.85 + 0.15 * Double(query.count) / Double(candidate.count))
        }
        if query.contains(candidate), candidate.count >= 2 {
            return 0.9 * Double(candidate.count) / Double(query.count)
        }

        let usesCJK = query.unicodeScalars.contains { scalar in
            scalar.value >= 0x4E00 && scalar.value <= 0x9FFF
        }
        let size = usesCJK || query.count < 5 ? 2 : 3
        let queryGrams = ngrams(query, size: size)
        guard !queryGrams.isEmpty else { return 0 }
        let coverage = Double(queryGrams.intersection(ngrams(candidate, size: size)).count)
            / Double(queryGrams.count)
        return coverage >= 0.30 ? 0.8 * coverage : 0
    }

    private static func compactText(_ value: String) -> String {
        value.precomposedStringWithCompatibilityMapping
            .lowercased()
            .unicodeScalars
            .filter { CharacterSet.alphanumerics.contains($0) }
            .map { String($0) }
            .joined()
    }

    private static func ngrams(_ value: String, size: Int) -> Set<String> {
        let characters = Array(value)
        guard !characters.isEmpty else { return [] }
        guard characters.count > size else { return [value] }
        return Set((0...(characters.count - size)).map { index in
            String(characters[index..<(index + size)])
        })
    }

    private static func structuredDateInterval(
        terms: [String],
        referenceDate: Date
    ) -> DateInterval? {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .autoupdatingCurrent
        let day = calendar.startOfDay(for: referenceDate)

        func interval(start: Date?, end: Date?) -> DateInterval? {
            guard let start, let end else { return nil }
            return DateInterval(start: start, end: end)
        }

        if terms.contains("今天") {
            return interval(start: day, end: calendar.date(byAdding: .day, value: 1, to: day))
        }
        if terms.contains("昨天") {
            return interval(start: calendar.date(byAdding: .day, value: -1, to: day), end: day)
        }
        if terms.contains("前天") {
            return interval(
                start: calendar.date(byAdding: .day, value: -2, to: day),
                end: calendar.date(byAdding: .day, value: -1, to: day)
            )
        }
        if terms.contains("明天") {
            return interval(
                start: calendar.date(byAdding: .day, value: 1, to: day),
                end: calendar.date(byAdding: .day, value: 2, to: day)
            )
        }
        if terms.contains("本周"), let week = calendar.dateInterval(of: .weekOfYear, for: day) {
            return week
        }
        if terms.contains("上周"),
           let currentWeek = calendar.dateInterval(of: .weekOfYear, for: day),
           let previousStart = calendar.date(byAdding: .weekOfYear, value: -1, to: currentWeek.start) {
            return DateInterval(start: previousStart, end: currentWeek.start)
        }
        if terms.contains("本月"), let month = calendar.dateInterval(of: .month, for: day) {
            return month
        }
        if terms.contains("上个月"),
           let currentMonth = calendar.dateInterval(of: .month, for: day),
           let previousStart = calendar.date(byAdding: .month, value: -1, to: currentMonth.start) {
            return DateInterval(start: previousStart, end: currentMonth.start)
        }
        if terms.contains("今年"), let year = calendar.dateInterval(of: .year, for: day) {
            return year
        }
        if terms.contains("去年"),
           let currentYear = calendar.dateInterval(of: .year, for: day),
           let previousStart = calendar.date(byAdding: .year, value: -1, to: currentYear.start) {
            return DateInterval(start: previousStart, end: currentYear.start)
        }
        return nil
    }
}
