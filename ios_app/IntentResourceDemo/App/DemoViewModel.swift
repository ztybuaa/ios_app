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
    @Published var inputText: String = "把小狗图片发给小明"
    @Published private(set) var inference: InferenceDisplay?
    @Published private(set) var searchResult: ResourceSearchResult?
    @Published private(set) var isRunning = false
    @Published private(set) var message: String?

    private let modelStore: ModelStore
    private let searchService = ResourceSearchService()

    init(modelStore: ModelStore) {
        self.modelStore = modelStore
    }

    func analyze() {
        let trimmed = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            message = DemoError.invalidInput.localizedDescription
            return
        }

        modelStore.loadIfNeeded()
        guard let model = modelStore.model else {
            message = modelStore.errorMessage ?? DemoError.modelNotLoaded.localizedDescription
            return
        }

        isRunning = true
        message = nil

        Task {
            let inferenceStart = CFAbsoluteTimeGetCurrent()
            let prediction = model.predict(trimmed)
            let inferenceTime = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1000
            let display = InferenceDisplay(
                input: trimmed,
                intent: prediction.intent,
                rawSlots: prediction.rawSlots,
                normalizedSlots: prediction.normalizedSlots,
                inferenceTimeMs: inferenceTime,
                loadTimeMs: modelStore.loadTimeMs,
                memoryMB: PerformanceMonitor.currentResidentMemoryMB()
            )

            let result = await searchService.search(
                intent: prediction.intent,
                slots: prediction.normalizedSlots
            )

            inference = display
            searchResult = result
            message = result.statusMessage
            isRunning = false
        }
    }
}
