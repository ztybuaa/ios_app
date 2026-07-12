import SwiftUI

struct InferenceView: View {
    let inference: InferenceDisplay

    var body: some View {
        Section("小模型输出") {
            MetricRow(label: "intent", value: inference.intent)
            MetricRow(label: "原始槽位", value: jsonString(inference.rawSlots))
            if let normalized = inference.normalizedSlots {
                MetricRow(label: "检索槽位", value: jsonString(normalized))
                MetricRow(label: "资源关键词", value: normalized.searchKeyword ?? "当前/泛指资源")
                MetricRow(label: "目标对象", value: normalized.targetKeyword)
            } else {
                MetricRow(label: "检索槽位", value: "{}")
            }
        }

        Section("性能") {
            MetricRow(label: "意图模型加载", value: String(format: "%.2f ms", inference.loadTimeMs))
            MetricRow(label: "意图识别推理", value: String(format: "%.2f ms", inference.inferenceTimeMs))
            MetricRow(label: "当前内存", value: inference.memoryMB.map { String(format: "%.2f MB", $0) } ?? "不可用")
        }
    }

    private func jsonString<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(value),
              let text = String(data: data, encoding: .utf8) else {
            return "-"
        }
        return text
    }
}
