import Foundation
import SwiftUI
import Translation

struct ContentView: View {
    @EnvironmentObject private var modelStore: ModelStore
    @StateObject private var viewModel: DemoViewModel
    @State private var translationDiagnosticIsBusy = false

    init() {
        _viewModel = StateObject(wrappedValue: DemoViewModel(modelStore: ModelStore.shared))
    }

    var body: some View {
        NavigationStack {
            List {
                inputSection
                if #available(iOS 18.0, *) {
                    TranslationDiagnosticSection(
                        isProductionRunning: viewModel.isRunning,
                        isBusy: $translationDiagnosticIsBusy
                    )
                }
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
                    .disabled(viewModel.isRunning || translationDiagnosticIsBusy)
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
            .disabled(viewModel.isRunning || translationDiagnosticIsBusy)

            if let error = modelStore.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
        }
    }
}

@available(iOS 18.0, *)
private struct TranslationDiagnosticSection: View {
    let isProductionRunning: Bool
    @Binding var isBusy: Bool

    @State private var showsSystemTranslation = false
    @State private var systemSourceText = "小猫图片"
    @State private var systemUIResult = "systemUI=notRun"
    @State private var replacementCallbackFired = false

    private var controlsDisabled: Bool {
        isProductionRunning || isBusy
    }

    var body: some View {
        Section("翻译对照") {
            AutomaticTranslationProbe(
                probeID: "nounPhrase",
                title: "自动会话：短语",
                sourceText: "小猫图片",
                systemImage: "text.bubble",
                isDisabled: controlsDisabled,
                onRunningChange: { isBusy = $0 }
            )

            AutomaticTranslationProbe(
                probeID: "fullCommand",
                title: "自动会话：完整句子",
                sourceText: "把小猫图片发给小明",
                systemImage: "text.quote",
                isDisabled: controlsDisabled,
                onRunningChange: { isBusy = $0 }
            )

            AutomaticTranslationProbe(
                probeID: "knownBaseline",
                title: "自动会话：基准句",
                sourceText: "你好，世界！",
                systemImage: "character.book.closed",
                isDisabled: controlsDisabled,
                onRunningChange: { isBusy = $0 }
            )

            Button {
                guard !controlsDisabled else {
                    return
                }
                systemSourceText = "小猫图片"
                replacementCallbackFired = false
                systemUIResult = [
                    "systemUI=requested",
                    "source=\(String(reflecting: systemSourceText))",
                    "replacementCallbackFired=false"
                ].joined(separator: "\n")
                isBusy = true
                showsSystemTranslation = true
            } label: {
                Label("系统翻译界面", systemImage: "globe")
            }
            .disabled(controlsDisabled)

            Text(systemUIResult)
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)

            Text(environmentResult)
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
        }
        .translationPresentation(
            isPresented: $showsSystemTranslation,
            text: systemSourceText
        ) { translatedText in
            replacementCallbackFired = true
            systemUIResult = [
                "systemUI=replacementReturned",
                "source=\(String(reflecting: systemSourceText))",
                "translated=\(String(reflecting: translatedText))",
                "characters=\(translatedText.count)",
                "utf8Bytes=\(translatedText.utf8.count)",
                "replacementCallbackFired=true"
            ].joined(separator: "\n")
        }
        .onChange(of: showsSystemTranslation) { isPresented in
            guard !isPresented else {
                return
            }
            if !replacementCallbackFired {
                systemUIResult = [
                    "systemUI=dismissedWithoutReplacement",
                    "source=\(String(reflecting: systemSourceText))",
                    "replacementCallbackFired=false"
                ].joined(separator: "\n")
            }
            isBusy = false
        }
        .onDisappear {
            isBusy = false
        }
    }

    private var environmentResult: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
        return [
            "appVersion=\(version)",
            "appBuild=\(build)",
            "os=\(ProcessInfo.processInfo.operatingSystemVersionString)",
            "preferredLanguages=\(Locale.preferredLanguages)"
        ].joined(separator: "\n")
    }
}

@available(iOS 18.0, *)
private struct AutomaticTranslationProbe: View {
    let probeID: String
    let title: String
    let sourceText: String
    let systemImage: String
    let isDisabled: Bool
    let onRunningChange: (Bool) -> Void

    @State private var configuration: TranslationSession.Configuration?
    @State private var attemptID: UUID?
    @State private var result = "notRun"

    var body: some View {
        Group {
            Button {
                run()
            } label: {
                Label(title, systemImage: systemImage)
            }
            .disabled(isDisabled || attemptID != nil)

            Text(result)
                .font(.footnote.monospaced())
                .foregroundStyle(.secondary)
        }
        .translationTask(configuration) { session in
            guard let currentAttemptID = attemptID else {
                return
            }
            let startedAt = Date()

            do {
                let response = try await session.translate(sourceText)
                let durationMs = Date().timeIntervalSince(startedAt) * 1_000
                let requestScalars = sourceText.unicodeScalars
                    .map { "U+\(String($0.value, radix: 16, uppercase: true))" }
                    .joined(separator: ", ")
                let targetScalars = response.targetText.unicodeScalars
                    .map { "U+\(String($0.value, radix: 16, uppercase: true))" }
                    .joined(separator: ", ")
                finish(
                    [
                        "probeID=\(probeID)",
                        "attemptID=\(currentAttemptID.uuidString)",
                        "api=single",
                        "configuredSource=nil",
                        "configuredTarget=nil",
                        "request=\(String(reflecting: sourceText))",
                        "requestCharacters=\(sourceText.count)",
                        "requestUTF8Bytes=\(sourceText.utf8.count)",
                        "requestScalars=[\(requestScalars)]",
                        "durationMs=\(String(format: "%.2f", durationMs))",
                        "responseSourceMatchesRequest=\(response.sourceText == sourceText)",
                        "sourceLanguage=\(String(reflecting: response.sourceLanguage))",
                        "targetLanguage=\(String(reflecting: response.targetLanguage))",
                        "rawTarget=\(String(reflecting: response.targetText))",
                        "targetCharacters=\(response.targetText.count)",
                        "targetUTF8Bytes=\(response.targetText.utf8.count)",
                        "targetScalars=[\(targetScalars)]"
                    ].joined(separator: "\n"),
                    attemptID: currentAttemptID
                )
            } catch {
                let durationMs = Date().timeIntervalSince(startedAt) * 1_000
                let nsError = error as NSError
                finish(
                    [
                        "probeID=\(probeID)",
                        "attemptID=\(currentAttemptID.uuidString)",
                        "api=single",
                        "configuredSource=nil",
                        "configuredTarget=nil",
                        "request=\(String(reflecting: sourceText))",
                        "durationMs=\(String(format: "%.2f", durationMs))",
                        "status=failed",
                        "type=\(String(reflecting: type(of: error)))",
                        "domain=\(nsError.domain)",
                        "code=\(nsError.code)",
                        "description=\(error.localizedDescription)"
                    ].joined(separator: "\n"),
                    attemptID: currentAttemptID
                )
            }
        }
        .onDisappear {
            guard attemptID != nil else {
                return
            }
            attemptID = nil
            onRunningChange(false)
        }
    }

    private func run() {
        guard attemptID == nil, !isDisabled else {
            return
        }
        let newAttemptID = UUID()
        attemptID = newAttemptID
        result = [
            "probeID=\(probeID)",
            "attemptID=\(newAttemptID.uuidString)",
            "status=running",
            "request=\(String(reflecting: sourceText))"
        ].joined(separator: "\n")
        onRunningChange(true)

        if configuration == nil {
            configuration = .init()
        } else {
            configuration?.invalidate()
        }
    }

    private func finish(_ output: String, attemptID completedAttemptID: UUID) {
        guard attemptID == completedAttemptID else {
            return
        }
        result = output
        attemptID = nil
        onRunningChange(false)
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
    static let availabilitySource = Locale.Language(identifier: "zh-Hans")
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
                        source: nil,
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
                    from: SemanticTranslationLanguages.availabilitySource,
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
                    let translationRequests = [
                        TranslationSession.Request(sourceText: request.sourceText)
                    ]
                    let responses = try await session.translations(from: translationRequests)

                    guard responses.count == 1, let response = responses.first else {
                        let diagnostic = [
                            "系统批量翻译没有返回唯一响应。",
                            "translationAPI=batch",
                            "configuredSource=automatic",
                            "availability=\(availabilityLabel)",
                            "request=\(String(reflecting: request.sourceText))",
                            "responseCount=\(responses.count)"
                        ].joined(separator: "\n")

                        await MainActor.run {
                            viewModel.failPendingTranslation(
                                id: request.id,
                                message: diagnostic,
                                stage: "停止：系统翻译响应数量异常"
                            )
                        }
                        return
                    }

                    let translatedText = response.targetText
                        .trimmingCharacters(in: .whitespacesAndNewlines)

                    guard !translatedText.isEmpty else {
                        let targetScalars = response.targetText.unicodeScalars
                            .map { "U+\(String($0.value, radix: 16, uppercase: true))" }
                            .joined(separator: ", ")
                        let diagnostic = [
                            "系统翻译返回空英文结果。",
                            "translationAPI=batch",
                            "configuredSource=automatic",
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
                        "translationAPI=batch",
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
