import Foundation

enum FeatureExtractor {
    private static let transferCues = [
        "发给", "发送给", "传给", "传送给", "转给", "分享给", "丢给",
        "给", "发到", "发送到", "传到"
    ]

    private static let actionCues = [
        "发", "发送", "传", "传送", "转", "转发", "分享", "丢"
    ]

    static func intentFeatures(_ text: String) -> [String: Double] {
        let chars = Array(text).map(String.init)
        var features: [String: Double] = [:]

        add(&features, "bias")
        add(&features, "len=\(lengthBucket(chars.count))")

        for n in 1...3 where chars.count >= n {
            for index in 0...(chars.count - n) {
                add(&features, "ng\(n):\(chars[index..<(index + n)].joined())")
            }
        }

        for prefixLength in 1...3 where chars.count >= prefixLength {
            add(&features, "prefix\(prefixLength):\(chars[0..<prefixLength].joined())")
            add(&features, "suffix\(prefixLength):\(chars[(chars.count - prefixLength)..<chars.count].joined())")
        }

        return features
    }

    static func spanFeatures(chars: [String], index: Int, previousTag: String) -> [String: Double] {
        let ch = token(chars, index)
        let prevCh = token(chars, index - 1)
        let nextCh = token(chars, index + 1)
        let prev2Ch = token(chars, index - 2)
        let next2Ch = token(chars, index + 2)
        let text = chars.joined()
        let leftContext = substring(chars, max(0, index - 10), index)
        let rightContext = substring(chars, index + 1, min(chars.count, index + 11))

        var features: [String: Double] = [:]
        add(&features, "bias")
        add(&features, "prev_tag=\(previousTag)")
        add(&features, "ch=\(ch)")
        add(&features, "prev_ch=\(prevCh)")
        add(&features, "next_ch=\(nextCh)")
        add(&features, "prev2_ch=\(prev2Ch)")
        add(&features, "next2_ch=\(next2Ch)")
        add(&features, "type=\(charType(ch))")
        add(&features, "prev_type=\(charType(prevCh))")
        add(&features, "next_type=\(charType(nextCh))")
        add(&features, "bigram_prev=\(prevCh)\(ch)")
        add(&features, "bigram_next=\(ch)\(nextCh)")
        add(&features, "trigram=\(prevCh)\(ch)\(nextCh)")

        for size in 1...5 {
            let left = substring(chars, max(0, index - size), index)
            let right = substring(chars, index + 1, min(chars.count, index + 1 + size))
            let currentPrefix = substring(chars, index, min(chars.count, index + size))
            let currentSuffix = substring(chars, max(0, index - size + 1), index + 1)
            add(&features, "left\(size)=\(left)")
            add(&features, "right\(size)=\(right)")
            add(&features, "prefix_at\(size)=\(currentPrefix)")
            add(&features, "suffix_at\(size)=\(currentSuffix)")
        }

        for cue in transferCues {
            if leftContext.hasSuffix(cue) { add(&features, "left_endswith_cue=\(cue)") }
            if leftContext.contains(cue) { add(&features, "left_has_cue=\(cue)") }
            if rightContext.hasPrefix(cue) { add(&features, "right_startswith_cue=\(cue)") }
            if rightContext.contains(cue) { add(&features, "right_has_cue=\(cue)") }
        }

        for cue in actionCues {
            if leftContext.hasSuffix(cue) { add(&features, "left_endswith_action=\(cue)") }
            if leftContext.contains(cue) { add(&features, "left_has_action=\(cue)") }
            if rightContext.hasPrefix(cue) { add(&features, "right_startswith_action=\(cue)") }
            if rightContext.contains(cue) { add(&features, "right_has_action=\(cue)") }
        }

        if leftContext.contains("给") && actionCues.contains(where: rightContext.contains) {
            add(&features, "between_give_and_action")
        }

        if index == 0 { add(&features, "is_start") }
        if index == chars.count - 1 { add(&features, "is_end") }

        let relativePosition = Int(10 * index / max(1, chars.count))
        add(&features, "pos_bucket=\(relativePosition)")

        _ = text
        return features
    }

    private static func add(_ features: inout [String: Double], _ name: String, value: Double = 1.0) {
        features[name, default: 0.0] += value
    }

    private static func lengthBucket(_ length: Int) -> String {
        if length <= 8 { return "short" }
        if length <= 16 { return "medium" }
        if length <= 28 { return "long" }
        return "very_long"
    }

    private static func token(_ chars: [String], _ index: Int) -> String {
        if index >= 0 && index < chars.count { return chars[index] }
        return index < 0 ? "<BOS>" : "<EOS>"
    }

    private static func substring(_ chars: [String], _ start: Int, _ end: Int) -> String {
        guard start < end, start >= 0, end <= chars.count else { return "" }
        return chars[start..<end].joined()
    }

    private static func charType(_ text: String) -> String {
        guard let scalar = text.unicodeScalars.first, text.count == 1 else {
            if text == "<BOS>" || text == "<EOS>" { return "other" }
            return "other"
        }

        if scalar.value >= 0x4E00 && scalar.value <= 0x9FFF { return "cjk" }
        if CharacterSet.decimalDigits.contains(scalar) { return "digit" }
        if scalar.isASCII && CharacterSet.letters.contains(scalar) { return "latin" }
        if CharacterSet.whitespacesAndNewlines.contains(scalar) { return "space" }
        if "_-（）()[]【】/\\.:：,，。;；".unicodeScalars.contains(scalar) { return "punct" }
        return "other"
    }
}
