import Foundation

struct InferenceDisplay {
    let input: String
    let intent: String
    let rawSlots: [String: String]
    let normalizedSlots: NormalizedSlots?
    let inferenceTimeMs: Double
    let loadTimeMs: Double
    let memoryMB: Double?
}

@MainActor
final class DemoViewModel: ObservableObject {
    @Published var inputText: String = "把小猫图片发给小明"
    @Published private(set) var inference: InferenceDisplay?
    @Published private(set) var searchResult: ResourceSearchResult?
    @Published private(set) var isRunning = false
    @Published private(set) var message: String?
    @Published private(set) var runStage: String

    private let modelStore: ModelStore
    private let searchService = ResourceSearchService()
    private static let runStageKey = "IntentResourceDemo.lastRunStage"

    init(modelStore: ModelStore) {
        self.modelStore = modelStore
        self.runStage = UserDefaults.standard.string(forKey: Self.runStageKey) ?? "尚未运行"
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
        inference = nil
        searchResult = nil
        markStage("模型已加载：准备推理")

        Task {
            let inferenceStart = CFAbsoluteTimeGetCurrent()
            let prediction = model.predict(trimmed)
            let inferenceTime = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1_000
            let display = InferenceDisplay(
                input: trimmed,
                intent: prediction.intent,
                rawSlots: prediction.rawSlots,
                normalizedSlots: prediction.normalizedSlots,
                inferenceTimeMs: inferenceTime,
                loadTimeMs: modelStore.loadTimeMs,
                memoryMB: PerformanceMonitor.currentResidentMemoryMB()
            )

            markStage("推理完成：准备资源候选检索")
            let result = await searchService.search(
                intent: prediction.intent,
                slots: prediction.normalizedSlots
            )
            markStage("检索完成：准备刷新界面")

            inference = display
            searchResult = result
            message = result.statusMessage
            isRunning = false
            markStage("完成：界面已刷新")
        }
    }

    private func markStage(_ stage: String) {
        runStage = stage
        UserDefaults.standard.set(stage, forKey: Self.runStageKey)
    }
}
