import Foundation

final class ChineseCLIPTokenizer {
    static let contextLength = 52
    static let vocabularySize = 21_128

    private let vocabulary: [String: Int32]
    private let paddingID: Int32
    private let unknownID: Int32
    private let classificationID: Int32
    private let separatorID: Int32

    init(bundle: Bundle = .main) throws {
        guard let url = bundle.url(forResource: "chinese_clip_vocab", withExtension: "txt") else {
            throw ChineseCLIPTokenizerError.missingVocabulary
        }

        let contents: String
        do {
            contents = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw ChineseCLIPTokenizerError.unreadableVocabulary(error.localizedDescription)
        }

        var tokens = contents.components(separatedBy: "\n")
        if tokens.last?.isEmpty == true {
            tokens.removeLast()
        }
        tokens = tokens.map { line in
            line.hasSuffix("\r") ? String(line.dropLast()) : line
        }
        guard tokens.count == Self.vocabularySize else {
            throw ChineseCLIPTokenizerError.invalidVocabulary(
                "expected \(Self.vocabularySize) tokens, found \(tokens.count)"
            )
        }

        var vocabulary: [String: Int32] = [:]
        vocabulary.reserveCapacity(tokens.count)
        for (index, token) in tokens.enumerated() {
            guard vocabulary.updateValue(Int32(index), forKey: token) == nil else {
                throw ChineseCLIPTokenizerError.invalidVocabulary("duplicate token \(token)")
            }
        }

        guard vocabulary["[PAD]"] == 0,
              vocabulary["[UNK]"] == 100,
              vocabulary["[CLS]"] == 101,
              vocabulary["[SEP]"] == 102 else {
            throw ChineseCLIPTokenizerError.invalidVocabulary("special-token IDs do not match Chinese-CLIP")
        }

        self.vocabulary = vocabulary
        self.paddingID = 0
        self.unknownID = 100
        self.classificationID = 101
        self.separatorID = 102
    }

    func encode(_ text: String) -> [Int32] {
        let tokenIDs = basicTokenize(text)
            .flatMap(wordPieceTokenize)
            .prefix(Self.contextLength - 2)
            .map { vocabulary[$0] ?? unknownID }

        var result = Array(repeating: paddingID, count: Self.contextLength)
        result[0] = classificationID
        for (index, tokenID) in tokenIDs.enumerated() {
            result[index + 1] = tokenID
        }
        result[tokenIDs.count + 1] = separatorID
        return result
    }

    private func basicTokenize(_ text: String) -> [String] {
        var initialTokens: [String] = []
        var current = ""

        func flushCurrent() {
            guard !current.isEmpty else { return }
            initialTokens.append(current)
            current.removeAll(keepingCapacity: true)
        }

        for scalar in text.unicodeScalars {
            if scalar.value == 0 || scalar.value == 0xFFFD || isControl(scalar) {
                continue
            }
            if isWhitespace(scalar) {
                flushCurrent()
            } else if isChineseCharacter(scalar.value) {
                flushCurrent()
                initialTokens.append(String(scalar))
            } else {
                current.unicodeScalars.append(scalar)
            }
        }
        flushCurrent()

        var output: [String] = []
        for initialToken in initialTokens {
            let normalized = stripAccents(initialToken.lowercased())
            var piece = ""
            for scalar in normalized.unicodeScalars {
                if isPunctuation(scalar) {
                    if !piece.isEmpty {
                        output.append(piece)
                        piece.removeAll(keepingCapacity: true)
                    }
                    output.append(String(scalar))
                } else {
                    piece.unicodeScalars.append(scalar)
                }
            }
            if !piece.isEmpty {
                output.append(piece)
            }
        }
        return output
    }

    private func wordPieceTokenize(_ token: String) -> [String] {
        let scalars = Array(token.unicodeScalars)
        guard scalars.count <= 200 else {
            return ["[UNK]"]
        }

        var result: [String] = []
        var start = 0
        while start < scalars.count {
            var end = scalars.count
            var match: String?
            while start < end {
                let value = scalarString(scalars[start..<end])
                let candidate = start == 0 ? value : "##\(value)"
                if vocabulary[candidate] != nil {
                    match = candidate
                    break
                }
                end -= 1
            }

            guard let match else {
                return ["[UNK]"]
            }
            result.append(match)
            start = end
        }
        return result
    }

    private func stripAccents(_ text: String) -> String {
        var result = ""
        for scalar in text.decomposedStringWithCanonicalMapping.unicodeScalars
            where scalar.properties.generalCategory != .nonspacingMark {
            result.unicodeScalars.append(scalar)
        }
        return result
    }

    private func scalarString(_ scalars: ArraySlice<Unicode.Scalar>) -> String {
        var result = ""
        for scalar in scalars {
            result.unicodeScalars.append(scalar)
        }
        return result
    }

    private func isWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        scalar == " " || scalar == "\t" || scalar == "\n" || scalar == "\r" ||
            scalar.properties.generalCategory == .spaceSeparator
    }

    private func isControl(_ scalar: Unicode.Scalar) -> Bool {
        if scalar == "\t" || scalar == "\n" || scalar == "\r" {
            return false
        }
        let category = scalar.properties.generalCategory
        return category == .control || category == .format
    }

    private func isPunctuation(_ scalar: Unicode.Scalar) -> Bool {
        let value = scalar.value
        if (33...47).contains(value) || (58...64).contains(value) ||
            (91...96).contains(value) || (123...126).contains(value) {
            return true
        }
        switch scalar.properties.generalCategory {
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
             .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return true
        default:
            return false
        }
    }

    private func isChineseCharacter(_ value: UInt32) -> Bool {
        (0x4E00...0x9FFF).contains(value) ||
            (0x3400...0x4DBF).contains(value) ||
            (0x20000...0x2A6DF).contains(value) ||
            (0x2A700...0x2B73F).contains(value) ||
            (0x2B740...0x2B81F).contains(value) ||
            (0x2B820...0x2CEAF).contains(value) ||
            (0xF900...0xFAFF).contains(value) ||
            (0x2F800...0x2FA1F).contains(value)
    }
}

private enum ChineseCLIPTokenizerError: LocalizedError {
    case missingVocabulary
    case unreadableVocabulary(String)
    case invalidVocabulary(String)

    var errorDescription: String? {
        switch self {
        case .missingVocabulary:
            return "Chinese-CLIP 词表资源 chinese_clip_vocab.txt 不存在。"
        case .unreadableVocabulary(let reason):
            return "Chinese-CLIP 词表无法读取：\(reason)"
        case .invalidVocabulary(let reason):
            return "Chinese-CLIP 词表不符合官方 RN50 契约：\(reason)。"
        }
    }
}
