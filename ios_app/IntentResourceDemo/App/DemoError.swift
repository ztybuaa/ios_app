import Foundation

enum DemoError: LocalizedError {
    case modelMissing
    case modelNotLoaded
    case permissionDenied(String)
    case resourceUnavailable(String)
    case invalidInput

    var errorDescription: String? {
        switch self {
        case .modelMissing:
            return "模型文件未找到，请确认 tiny_intent_slot_model.json 已加入 App Bundle。"
        case .modelNotLoaded:
            return "模型尚未加载。"
        case .permissionDenied(let message):
            return message
        case .resourceUnavailable(let message):
            return message
        case .invalidInput:
            return "请输入自然语言指令。"
        }
    }
}
