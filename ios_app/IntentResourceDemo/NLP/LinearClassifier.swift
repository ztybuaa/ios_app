import Foundation

struct LinearClassifierPayload: Decodable {
    let labels: [String]
    let weights: [String: [String: Double]]
}

struct LinearClassifier {
    let labels: [String]
    let weights: [String: [String: Double]]

    init(payload: LinearClassifierPayload) {
        self.labels = payload.labels
        self.weights = payload.weights
    }

    func predict(features: [String: Double]) -> String {
        var scores = Dictionary(uniqueKeysWithValues: labels.map { ($0, 0.0) })

        for (feature, value) in features {
            guard let labelWeights = weights[feature] else { continue }
            for (label, weight) in labelWeights {
                scores[label, default: 0.0] += weight * value
            }
        }

        var bestLabel = labels.first ?? ""
        var bestScore = scores[bestLabel] ?? 0.0

        for label in labels {
            let score = scores[label] ?? 0.0
            if score > bestScore || (score == bestScore && label > bestLabel) {
                bestLabel = label
                bestScore = score
            }
        }

        return bestLabel
    }
}
