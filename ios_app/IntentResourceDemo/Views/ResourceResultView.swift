import Contacts
import ContactsUI
import Photos
import SwiftUI
import UIKit

struct ResourceResultView: View {
    let result: ResourceSearchResult

    var body: some View {
        Section("资源模块") {
            MetricRow(label: "模块", value: result.moduleName)
            MetricRow(label: "端到端检索", value: duration(result.searchTimeMs))
            MetricRow(label: "当前内存", value: result.memoryMB.map { String(format: "%.2f MB", $0) } ?? "不可用")
        }

        if let metrics = result.semanticMetrics {
            Section("RN50 语义检索") {
                MetricRow(label: "模型加载", value: duration(metrics.modelLoadTimeMs))
                MetricRow(label: "语义总耗时", value: duration(metrics.totalWallMs))
                MetricRow(label: "快速粗排", value: duration(metrics.fullPassWallMs))
                MetricRow(label: "高质量终排", value: duration(metrics.rerankWallMs))
                MetricRow(
                    label: "扫描 / 重排",
                    value: "\(metrics.scannedAssetCount) / \(metrics.shortlistAssetCount) 张"
                )
                MetricRow(
                    label: "图像推理",
                    value: "粗排 \(metrics.coarsePredictionCount)，终排 \(metrics.qualityPredictionCount)"
                )
                MetricRow(
                    label: "缓存命中",
                    value: "内存 \(metrics.memoryCacheHits)，磁盘 \(metrics.diskCacheHits)"
                )
                MetricRow(
                    label: "缓存未命中",
                    value: "\(metrics.cacheMisses)"
                )
                if metrics.imageRequestFailures > 0 ||
                    metrics.corruptCacheEvictions > 0 ||
                    metrics.cacheStorageUnavailable > 0 {
                    MetricRow(
                        label: "异常计数",
                        value: "图片 \(metrics.imageRequestFailures)，损坏缓存 \(metrics.corruptCacheEvictions)，存储 \(metrics.cacheStorageUnavailable)"
                    )
                }
            }
        }

        CandidateSection(title: "资源候选", candidates: result.resourceCandidates)
        CandidateSection(title: "目标候选", candidates: result.targetCandidates)
    }

    private func duration(_ milliseconds: Double) -> String {
        if milliseconds >= 1_000 {
            return String(format: "%.2f s", milliseconds / 1_000)
        }
        return String(format: "%.2f ms", milliseconds)
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
                    NavigationLink {
                        CandidateDetailView(candidate: candidate)
                    } label: {
                        CandidateRow(candidate: candidate)
                    }
                }
            }
        }
    }
}

private struct CandidateRow: View {
    let candidate: ResourceCandidate

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: iconName(candidate.kind))
                    .foregroundStyle(.tint)
                    .frame(width: 24)
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
                .lineLimit(2)
        }
        .padding(.vertical, 4)
    }
}

private struct CandidateDetailView: View {
    let candidate: ResourceCandidate
    @State private var contactToShow: ContactSheetItem?
    @State private var contactError: String?

    var body: some View {
        List {
            if candidate.kind == .photo || candidate.kind == .video {
                Section {
                    AssetPreviewView(assetIdentifier: candidate.id, kind: candidate.kind)
                }
            }

            Section("候选详情") {
                MetricRow(label: "名称", value: candidate.title)
                MetricRow(label: "类型", value: kindTitle(candidate.kind))
                MetricRow(label: "位置", value: candidate.subtitle.isEmpty ? "无" : candidate.subtitle)
                MetricRow(label: "得分", value: String(format: "%.2f", candidate.score))
                MetricRow(label: "标识", value: candidate.id)
            }

            if !candidate.detail.isEmpty {
                Section("描述") {
                    Text(candidate.detail)
                        .font(.body)
                }
            }

            Section("匹配信息") {
                Text(candidate.debugInfo)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("操作") {
                if candidate.kind == .contact {
                    Button {
                        openContact()
                    } label: {
                        Label("打开联系人卡片", systemImage: "person.text.rectangle")
                    }
                }

                ShareLink(item: shareText) {
                    Label("分享候选信息", systemImage: "square.and.arrow.up")
                }

                if let contactError {
                    Text(contactError)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
        }
        .navigationTitle(candidate.title)
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $contactToShow) { item in
            ContactCardView(contact: item.contact)
        }
    }

    private var shareText: String {
        [
            candidate.title,
            candidate.subtitle,
            candidate.detail,
            "id: \(candidate.id)",
            "score: \(String(format: "%.2f", candidate.score))"
        ]
        .filter { !$0.isEmpty }
        .joined(separator: "\n")
    }

    private func openContact() {
        do {
            let descriptor = CNContactViewController.descriptorForRequiredKeys()
            let contact = try CNContactStore().unifiedContact(withIdentifier: candidate.id, keysToFetch: [descriptor])
            contactError = nil
            contactToShow = ContactSheetItem(contact: contact)
        } catch {
            contactError = "无法打开系统联系人卡片：\(error.localizedDescription)"
        }
    }
}

private struct ContactSheetItem: Identifiable {
    let contact: CNContact

    var id: String {
        contact.identifier
    }
}

private struct AssetPreviewView: View {
    let assetIdentifier: String
    let kind: CandidateKind

    @State private var image: UIImage?
    @State private var message = "正在加载预览..."

    var body: some View {
        Group {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity)
                    .frame(height: 240)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .accessibilityLabel(kind == .video ? "视频预览" : "照片预览")
            } else {
                VStack(spacing: 10) {
                    Image(systemName: kind == .video ? "video" : "photo")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text(message)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
                .frame(height: 180)
            }
        }
        .task(id: assetIdentifier) {
            loadPreview()
        }
    }

    private func loadPreview() {
        let assets = PHAsset.fetchAssets(withLocalIdentifiers: [assetIdentifier], options: nil)
        guard let asset = assets.firstObject else {
            message = "未找到本机相册资源。"
            return
        }

        let options = PHImageRequestOptions()
        options.deliveryMode = .opportunistic
        options.resizeMode = .fast
        options.isNetworkAccessAllowed = false

        PHImageManager.default().requestImage(
            for: asset,
            targetSize: CGSize(width: 640, height: 640),
            contentMode: .aspectFit,
            options: options
        ) { newImage, info in
            DispatchQueue.main.async {
                if let newImage {
                    image = newImage
                    message = ""
                    return
                }
                if info?[PHImageErrorKey] != nil {
                    message = "预览读取失败。"
                }
            }
        }
    }
}

private struct ContactCardView: UIViewControllerRepresentable {
    let contact: CNContact

    func makeUIViewController(context: Context) -> UINavigationController {
        let controller = CNContactViewController(for: contact)
        controller.allowsActions = true
        controller.allowsEditing = false
        return UINavigationController(rootViewController: controller)
    }

    func updateUIViewController(_ uiViewController: UINavigationController, context: Context) {}
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

private func kindTitle(_ kind: CandidateKind) -> String {
    switch kind {
    case .photo: return "照片"
    case .video: return "视频"
    case .file: return "文件"
    case .folder: return "文件夹"
    case .contact: return "联系人"
    }
}
