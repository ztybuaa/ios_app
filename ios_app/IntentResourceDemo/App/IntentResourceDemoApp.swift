import SwiftUI

@main
struct IntentResourceDemoApp: App {
    @StateObject private var modelStore = ModelStore.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelStore)
                .onAppear {
                    modelStore.loadIfNeeded()
                }
        }
    }
}
