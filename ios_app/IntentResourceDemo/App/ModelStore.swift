import Foundation

@MainActor
final class ModelStore: ObservableObject {
    static let shared = ModelStore()

    @Published private(set) var model: TinyIntentSlotModel?
    @Published private(set) var loadTimeMs: Double = 0
    @Published private(set) var errorMessage: String?

    private init() {}

    func loadIfNeeded() {
        guard model == nil else { return }

        let start = CFAbsoluteTimeGetCurrent()
        do {
            model = try TinyIntentSlotModel.loadFromBundle()
            loadTimeMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
            errorMessage = nil
        } catch {
            errorMessage = "模型加载失败：\(error.localizedDescription)"
        }
    }
}
