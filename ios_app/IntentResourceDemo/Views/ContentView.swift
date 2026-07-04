import SwiftUI

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
