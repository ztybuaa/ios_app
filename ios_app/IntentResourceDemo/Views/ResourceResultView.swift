import SwiftUI

struct ResourceResultView: View {
    let result: ResourceSearchResult

    var body: some View {
        Section("资源模块") {
            MetricRow(label: "模块", value: result.moduleName)
            MetricRow(label: "检索耗时", value: String(format: "%.2f ms", result.searchTimeMs))
            MetricRow(label: "当前内存", value: result.memoryMB.map { String(format: "%.2f MB", $0) } ?? "不可用")
        }

        CandidateSection(title: "资源候选", candidates: result.resourceCandidates)
        CandidateSection(title: "目标候选", candidates: result.targetCandidates)
    }
}

struct CandidateSection: View {
    let title: String
    let candidates: [ResourceCandidate]

    var body: some View {
        Section(title) {
            if candidates.isEmpty {
                Text("暂无候选")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(candidates) { candidate in
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Image(systemName: iconName(candidate.kind))
                                .foregroundStyle(.tint)
                            Text(candidate.title)
                                .font(.headline)
                            Spacer()
                            Text(String(format: "%.1f", candidate.score))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Text(candidate.subtitle)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        if !candidate.detail.isEmpty {
                            Text(candidate.detail)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                        Text(candidate.debugInfo)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(.vertical, 4)
                }
            }
        }
    }

    private func iconName(_ kind: CandidateKind) -> String {
        switch kind {
        case .photo: return "photo"
        case .video: return "video"
        case .file: return "doc"
        case .folder: return "folder"
        case .contact: return "person.crop.circle"
        }
    }
}
