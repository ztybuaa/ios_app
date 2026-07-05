import Foundation

struct InferenceDisplay {
    let input: String
    let intent: String
    let rawSlots: [String: String]
    let normalizedSlots: NormalizedSlots?
    let semanticQueryText: String?
    let inferenceTimeMs: Double
    let loadTimeMs: Double
    let memoryMB: Double?
}

struct SemanticTranslationRequest: Equatable {
    let id: UUID
    let sourceText: String
}

@MainActor
final class DemoViewModel: ObservableObject {
    @Published var inputText: String = "把小猫图片发给小明"
    @Published private(set) var inference: InferenceDisplay?
    @Published private(set) var searchResult: ResourceSearchResult?
    @Published private(set) var isRunning = false
    @Published private(set) var message: String?
    @Published private(set) var runStage: String
    @Published private(set) var pendingTranslationRequest: SemanticTranslationRequest?

    var pendingTranslationRequestID: UUID? {
        pendingTranslationRequest?.id
    }

    private let modelStore: ModelStore
    private let searchService = ResourceSearchService()
    private var pendingSearch: PendingSemanticSearch?
    private static let runStageKey = "IntentResourceDemo.lastRunStage"

    init(modelStore: ModelStore) {
        self.modelStore = modelStore
        self.runStage = UserDefaults.standard.string(forKey: DemoViewModel.runStageKey) ?? "尚未运行"
    }

    func analyze() {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            message = DemoError.invalidInput.localizedDescription
            return
        }

        markStage("开始：准备加载模型")
        modelStore.loadIfNeeded()
        guard let model = modelStore.model else {
            message = modelStore.errorMessage ?? DemoError.modelNotLoaded.localizedDescription
            markStage("停止：模型加载失败")
            return
        }

        isRunning = true
        message = nil
        markStage("模型已加载：准备推理")

        Task {
            let inferenceStart = CFAbsoluteTimeGetCurrent()
            let prediction = model.predict(trimmed)
            let inferenceTime = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1000
            markStage("推理完成：准备检索候选")
            let display = InferenceDisplay(
                input: trimmed,
                intent: prediction.intent,
                rawSlots: prediction.rawSlots,
                normalizedSlots: prediction.normalizedSlots,
                semanticQueryText: nil,
                inferenceTimeMs: inferenceTime,
                loadTimeMs: modelStore.loadTimeMs,
                memoryMB: PerformanceMonitor.currentResidentMemoryMB()
            )

            if let slots = prediction.normalizedSlots,
               shouldTranslateBeforeSemanticSearch(intent: prediction.intent, slots: slots) {
                let request = SemanticTranslationRequest(id: UUID(), sourceText: semanticSourceText(from: slots))
                pendingSearch = PendingSemanticSearch(
                    requestID: request.id,
                    intent: prediction.intent,
                    slots: slots,
                    display: display
                )
                inference = display
                searchResult = nil
                pendingTranslationRequest = request
                message = "正在调用系统翻译，把中文需求转成 MobileCLIP 的英文语义查询。"
                markStage("等待系统翻译：\(request.sourceText)")
                return
            }

            let result = await searchService.search(
                intent: prediction.intent,
                slots: prediction.normalizedSlots,
                semanticQueryText: nil
            )
            markStage("检索完成：准备刷新界面")

            inference = display
            searchResult = result
            message = result.statusMessage
            isRunning = false
            markStage("完成：界面已刷新")
        }
    }

    func completePendingTranslation(id: UUID, translatedText: String) {
        guard let pendingSearch,
              pendingSearch.requestID == id,
              pendingTranslationRequest?.id == id else {
            return
        }

        let semanticQuery = translatedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !semanticQuery.isEmpty else {
            failPendingTranslation(id: id, message: "系统翻译没有返回有效英文查询，已停止检索。")
            return
        }

        pendingTranslationRequest = nil
        self.pendingSearch = nil
        message = "系统翻译完成：\(semanticQuery)"
        markStage("翻译完成：准备语义检索")

        let display = InferenceDisplay(
            input: pendingSearch.display.input,
            intent: pendingSearch.display.intent,
            rawSlots: pendingSearch.display.rawSlots,
            normalizedSlots: pendingSearch.display.normalizedSlots,
            semanticQueryText: semanticQuery,
            inferenceTimeMs: pendingSearch.display.inferenceTimeMs,
            loadTimeMs: pendingSearch.display.loadTimeMs,
            memoryMB: pendingSearch.display.memoryMB
        )

        Task {
            let result = await searchService.search(
                intent: pendingSearch.intent,
                slots: pendingSearch.slots,
                semanticQueryText: semanticQuery
            )
            markStage("检索完成：准备刷新界面")
            inference = display
            searchResult = result
            message = result.statusMessage
            isRunning = false
            markStage("完成：界面已刷新")
        }
    }

    func failPendingTranslation(id: UUID?, message: String) {
        guard id == nil || pendingTranslationRequest?.id == id else {
            return
        }
        pendingTranslationRequest = nil
        pendingSearch = nil
        self.message = message
        isRunning = false
        markStage("停止：系统翻译不可用")
    }

    private func markStage(_ stage: String) {
        runStage = stage
        UserDefaults.standard.set(stage, forKey: Self.runStageKey)
    }

    private func shouldTranslateBeforeSemanticSearch(intent: String, slots: NormalizedSlots) -> Bool {
        guard intent == "photo" else {
            return false
        }
        let sourceText = semanticSourceText(from: slots)
        return !sourceText.isEmpty && containsCJK(sourceText)
    }

    private func semanticSourceText(from slots: NormalizedSlots) -> String {
        [slots.searchKeyword, slots.resourcePhrase]
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    private func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            (0x4E00...0x9FFF).contains(Int(scalar.value))
        }
    }
}

private struct PendingSemanticSearch {
    let requestID: UUID
    let intent: String
    let slots: NormalizedSlots
    let display: InferenceDisplay
}
