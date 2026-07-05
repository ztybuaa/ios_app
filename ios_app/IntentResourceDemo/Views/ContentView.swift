import SwiftUI
import Translation

struct ContentView: View {
    @EnvironmentObject private var modelStore: ModelStore
    @StateObject private var viewModel: DemoViewModel

    init() {
        _viewModel = StateObject(wrappedValue: DemoViewModel(modelStore: ModelStore.shared))
    }

    var body: some View {
        NavigationStack {
            List {
                inputSection
                if let inference = viewModel.inference {
                    InferenceView(inference: inference)
                }
                if let result = viewModel.searchResult {
                    ResourceResultView(result: result)
                }
                Section("诊断") {
                    Text(viewModel.runStage)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                if let message = viewModel.message {
                    Section("状态") {
                        Text(message)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("端侧资源 Demo")
            .semanticTranslationBridge(viewModel: viewModel)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        viewModel.analyze()
                    } label: {
                        if viewModel.isRunning {
                            ProgressView()
                        } else {
                            Label("运行", systemImage: "play.fill")
                        }
                    }
                    .disabled(viewModel.isRunning)
                }
            }
        }
    }

    private var inputSection: some View {
        Section("自然语言输入") {
            TextField("输入一句资源分享指令", text: $viewModel.inputText, axis: .vertical)
                .lineLimit(2...4)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()

            Button {
                viewModel.analyze()
            } label: {
                Label("分析并检索候选", systemImage: "magnifyingglass")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.isRunning)

            if let error = modelStore.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
        }
    }
}

private extension View {
    @ViewBuilder
    func semanticTranslationBridge(viewModel: DemoViewModel) -> some View {
        if #available(iOS 18.0, *) {
            modifier(SemanticTranslationBridge(viewModel: viewModel))
        } else {
            modifier(UnsupportedSemanticTranslationBridge(viewModel: viewModel))
        }
    }
}

@available(iOS 18.0, *)
private struct SemanticTranslationBridge: ViewModifier {
    @ObservedObject var viewModel: DemoViewModel
    @State private var configuration: TranslationSession.Configuration?

    func body(content: Content) -> some View {
        content
            .onChange(of: viewModel.pendingTranslationRequestID) { _ in
                guard viewModel.pendingTranslationRequest != nil else {
                    return
                }
                if configuration == nil {
                    configuration = TranslationSession.Configuration(
                        source: Locale.Language(identifier: "zh-Hans"),
                        target: Locale.Language(identifier: "en")
                    )
                } else {
                    configuration?.invalidate()
                }
            }
            .translationTask(configuration) { session in
                guard let request = await MainActor.run(body: { viewModel.pendingTranslationRequest }) else {
                    return
                }
                do {
                    try await session.prepareTranslation()
                    let response = try await session.translate(request.sourceText)
                    await MainActor.run {
                        viewModel.completePendingTranslation(id: request.id, translatedText: response.targetText)
                    }
                } catch {
                    await MainActor.run {
                        viewModel.failPendingTranslation(
                            id: request.id,
                            message: "系统翻译失败：\(error.localizedDescription)。请确认设备已支持并下载中英翻译语言包。"
                        )
                    }
                }
            }
    }
}

private struct UnsupportedSemanticTranslationBridge: ViewModifier {
    @ObservedObject var viewModel: DemoViewModel

    func body(content: Content) -> some View {
        content
            .onChange(of: viewModel.pendingTranslationRequestID) { requestID in
                guard requestID != nil else {
                    return
                }
                viewModel.failPendingTranslation(
                    id: requestID,
                    message: "当前 iOS 版本不支持程序化系统翻译。请升级到 iOS 18 或更高版本，或直接输入英文图片描述。"
                )
            }
    }
}
