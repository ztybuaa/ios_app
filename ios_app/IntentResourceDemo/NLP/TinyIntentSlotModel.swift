import Foundation

struct TinyModelPayload: Decodable {
    let version: Int
    let intentModel: LinearClassifierPayload
    let contentModel: LinearClassifierPayload
    let targetModel: LinearClassifierPayload

    enum CodingKeys: String, CodingKey {
        case version
        case intentModel = "intent_model"
        case contentModel = "content_model"
        case targetModel = "target_model"
    }
}

struct TinyPrediction {
    let intent: String
    let rawSlots: [String: String]
    let normalizedSlots: NormalizedSlots?
}

final class TinyIntentSlotModel {
    private let intentClassifier: LinearClassifier
    private let contentClassifier: LinearClassifier
    private let targetClassifier: LinearClassifier

    init(payload: TinyModelPayload) {
        self.intentClassifier = LinearClassifier(payload: payload.intentModel)
        self.contentClassifier = LinearClassifier(payload: payload.contentModel)
        self.targetClassifier = LinearClassifier(payload: payload.targetModel)
    }

    static func loadFromBundle() throws -> TinyIntentSlotModel {
        guard let url = Bundle.main.url(forResource: "tiny_intent_slot_model", withExtension: "json") else {
            throw DemoError.modelMissing
        }
        let data = try Data(contentsOf: url)
        let payload = try JSONDecoder().decode(TinyModelPayload.self, from: data)
        return TinyIntentSlotModel(payload: payload)
    }

    func predict(_ text: String) -> TinyPrediction {
        let intent = intentClassifier.predict(features: FeatureExtractor.intentFeatures(text))
        let chars = Array(text).map(String.init)
        let contentTags = predictSpanTags(chars: chars, classifier: contentClassifier)
        let targetTags = predictSpanTags(chars: chars, classifier: targetClassifier)

        var slots: [String: String] = [:]
        if intent != "unknown" {
            if let content = spanText(chars: chars, tags: contentTags) {
                slots["share_content"] = content
            }
            if let target = spanText(chars: chars, tags: targetTags) {
                slots["share_target"] = target
            }
        }

        return TinyPrediction(
            intent: intent,
            rawSlots: slots,
            normalizedSlots: SlotNormalizer.normalize(intent: intent, rawSlots: slots)
        )
    }

    private func predictSpanTags(chars: [String], classifier: LinearClassifier) -> [String] {
        var tags: [String] = []
        var previousTag = "<START>"

        for index in chars.indices {
            var tag = classifier.predict(features: FeatureExtractor.spanFeatures(chars: chars, index: index, previousTag: previousTag))
            if tag == "I" && previousTag != "B" && previousTag != "I" {
                tag = "B"
            }
            tags.append(tag)
            previousTag = tag
        }

        return tags
    }

    private func spanText(chars: [String], tags: [String]) -> String? {
        var index = 0
        while index < tags.count {
            if tags[index] == "B" {
                let start = index
                index += 1
                while index < tags.count && tags[index] == "I" {
                    index += 1
                }
                return chars[start..<index].joined()
            }
            index += 1
        }
        return nil
    }
}
