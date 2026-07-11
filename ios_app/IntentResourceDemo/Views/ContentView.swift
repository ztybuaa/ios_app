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
private enum SemanticTranslationLanguages {
    static let source = Locale.Language(identifier: "zh-Hans")
    static let target = Locale.Language(identifier: "en-US")

    static func label(for status: LanguageAvailability.Status) -> String {
        switch status {
        case .installed:
            return "installed"
        case .supported:
            return "supported"
        case .unsupported:
            return "unsupported"
        @unknown default:
            return "unknown"
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
                        source: SemanticTranslationLanguages.source,
                        target: SemanticTranslationLanguages.target
                    )
                } else {
                    configuration?.invalidate()
                }
            }
            .translationTask(configuration) { session in
                guard let request = await MainActor.run(body: { viewModel.pendingTranslationRequest }) else {
                    return
                }

                let availability = await LanguageAvailability().status(
                    from: SemanticTranslationLanguages.source,
                    to: SemanticTranslationLanguages.target
                )
                let availabilityLabel = SemanticTranslationLanguages.label(for: availability)

                switch availability {
                case .installed, .supported:
                    break
                case .unsupported:
                    await MainActor.run {
                        viewModel.failPendingTranslation(
                            id: request.id,
                            message: "系统翻译不支持 zh-Hans → en-US。",
                            stage: "停止：翻译语言组合不受支持"
                        )
                    }
                    return
                @unknown default:
                    await MainActor.run {
                        viewModel.failPendingTranslation(
                            id: request.id,
                            message: "系统返回未知翻译可用性状态。",
                            stage: "停止：未知翻译可用性状态"
                        )
                    }
                    return
                }

                do {
                    let response = try await session.translate(request.sourceText)
                    let translatedText = response.targetText
                        .trimmingCharacters(in: .whitespacesAndNewlines)

                    guard !translatedText.isEmpty else {
                        let targetScalars = response.targetText.unicodeScalars
                            .map { "U+\(String($0.value, radix: 16, uppercase: true))" }
                            .joined(separator: ", ")
                        let diagnostic = [
                            "系统翻译返回空英文结果。",
                            "availability=\(availabilityLabel)",
                            "request=\(String(reflecting: request.sourceText))",
                            "responseSource=\(String(reflecting: response.sourceText))",
                            "sourceLanguage=\(String(reflecting: response.sourceLanguage))",
                            "targetLanguage=\(String(reflecting: response.targetLanguage))",
                            "rawTarget=\(String(reflecting: response.targetText))",
                            "targetCharacters=\(response.targetText.count)",
                            "targetUTF8Bytes=\(response.targetText.utf8.count)",
                            "targetScalars=[\(targetScalars)]"
                        ].joined(separator: "\n")

                        await MainActor.run {
                            viewModel.failPendingTranslation(
                                id: request.id,
                                message: diagnostic,
                                stage: "停止：系统翻译返回空响应"
                            )
                        }
                        return
                    }

                    await MainActor.run {
                        viewModel.completePendingTranslation(
                            id: request.id,
                            translatedText: translatedText
                        )
                    }
                } catch {
                    let nsError = error as NSError
                    let diagnostic = [
                        "系统翻译失败。",
                        "type=\(String(reflecting: type(of: error)))",
                        "availability=\(availabilityLabel)",
                        "domain=\(nsError.domain)",
                        "code=\(nsError.code)",
                        "description=\(error.localizedDescription)"
                    ].joined(separator: "\n")

                    await MainActor.run {
                        viewModel.failPendingTranslation(
                            id: request.id,
                            message: diagnostic
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
