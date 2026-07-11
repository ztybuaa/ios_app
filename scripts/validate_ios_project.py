import json
import plistlib
import re
from pathlib import Path

from env_guard import ensure_project_venv


ensure_project_venv()


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = ROOT / "ios_app" / "IntentResourceDemo"
PROJECT_FILE = ROOT / "ios_app" / "IntentResourceDemo.xcodeproj" / "project.pbxproj"
SCHEME_FILE = (
    ROOT
    / "ios_app"
    / "IntentResourceDemo.xcodeproj"
    / "xcshareddata"
    / "xcschemes"
    / "IntentResourceDemo.xcscheme"
)
GITHUB_WORKFLOW = ROOT / ".github" / "workflows" / "build-ios-unsigned-ipa.yml"
CI_BUILD_SCRIPT = ROOT / "scripts" / "ci" / "build_unsigned_ipa.sh"
DEMO_VIEW_MODEL = APP_ROOT / "App" / "DemoViewModel.swift"
CONTENT_VIEW = APP_ROOT / "Views" / "ContentView.swift"

SWIFT_FILES = [
    "App/DemoError.swift",
    "App/DemoViewModel.swift",
    "App/IntentResourceDemoApp.swift",
    "App/ModelStore.swift",
    "App/PerformanceMonitor.swift",
    "NLP/FeatureExtractor.swift",
    "NLP/LinearClassifier.swift",
    "NLP/SlotNormalizer.swift",
    "NLP/TinyIntentSlotModel.swift",
    "NLP/Tokenizer/CLIPTokenizer.swift",
    "NLP/Tokenizer/GPT2ByteEncoder.swift",
    "NLP/Tokenizer/Utils.swift",
    "ResourceModules/ContactResourceModule.swift",
    "ResourceModules/FileFolderResourceModule.swift",
    "ResourceModules/MediaResourceModule.swift",
    "ResourceModules/ResourceModels.swift",
    "ResourceModules/ResourceSearchService.swift",
    "ResourceModules/SemanticImageSearchService.swift",
    "Views/ContentView.swift",
    "Views/InferenceView.swift",
    "Views/MetricRow.swift",
    "Views/ResourceResultView.swift",
]

RESOURCE_FILES = [
    "Resources/tiny_intent_slot_model.json",
    "Resources/sample_resource_index.json",
    "Resources/clip-vocab.json",
    "Resources/clip-merges.txt",
]

MOBILECLIP_FILES = [
    "Resources/MobileCLIP/mobileclip_s0_image.mlpackage/Manifest.json",
    "Resources/MobileCLIP/mobileclip_s0_image.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "Resources/MobileCLIP/mobileclip_s0_image.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "Resources/MobileCLIP/mobileclip_s0_text.mlpackage/Manifest.json",
    "Resources/MobileCLIP/mobileclip_s0_text.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "Resources/MobileCLIP/mobileclip_s0_text.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
]


def require(condition, message):
    if not condition:
        raise SystemExit(f"VALIDATION FAILED: {message}")


def validate_files():
    require(PROJECT_FILE.exists(), f"missing {PROJECT_FILE}")
    require(SCHEME_FILE.exists(), f"missing shared scheme {SCHEME_FILE}")
    require(GITHUB_WORKFLOW.exists(), f"missing GitHub Actions workflow {GITHUB_WORKFLOW}")
    require(CI_BUILD_SCRIPT.exists(), f"missing CI build script {CI_BUILD_SCRIPT}")
    for relative in SWIFT_FILES + RESOURCE_FILES + MOBILECLIP_FILES + ["Support/Info.plist"]:
        require((APP_ROOT / relative).exists(), f"missing app file {relative}")


def validate_model():
    model_path = APP_ROOT / "Resources" / "tiny_intent_slot_model.json"
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    for key in ("version", "intent_model", "content_model", "target_model"):
        require(key in payload, f"model missing key {key}")
    for key in ("intent_model", "content_model", "target_model"):
        require("labels" in payload[key], f"{key} missing labels")
        require("weights" in payload[key], f"{key} missing weights")


def validate_resource_index():
    index_path = APP_ROOT / "Resources" / "sample_resource_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    require(payload.get("files"), "sample resource index has no files")
    require(payload.get("folders"), "sample resource index has no folders")
    for section in ("files", "folders"):
        for item in payload[section]:
            for key in ("id", "kind", "title", "path", "summary", "tags"):
                require(key in item, f"{section} item missing {key}")


def validate_info_plist():
    plist_path = APP_ROOT / "Support" / "Info.plist"
    with plist_path.open("rb") as f:
        payload = plistlib.load(f)
    require("NSContactsUsageDescription" in payload, "Info.plist missing contacts usage description")
    require("NSPhotoLibraryUsageDescription" in payload, "Info.plist missing photo usage description")
    require(payload.get("LSRequiresIPhoneOS") is True, "Info.plist should require iPhone OS")
    require(payload.get("CFBundleShortVersionString") == "$(MARKETING_VERSION)", "app version should use build setting")
    require(payload.get("CFBundleVersion") == "$(CURRENT_PROJECT_VERSION)", "build number should use build setting")


def validate_project_references():
    project_text = PROJECT_FILE.read_text(encoding="utf-8")
    for relative in SWIFT_FILES + RESOURCE_FILES:
        name = Path(relative).name
        require(name in project_text, f"project.pbxproj missing reference to {name}")
    for package_name in ("mobileclip_s0_image.mlpackage", "mobileclip_s0_text.mlpackage"):
        require(package_name in project_text, f"project.pbxproj missing reference to {package_name}")
        require(f"{package_name} in Sources" in project_text, f"{package_name} should be in Sources for Core ML codegen")
    require("PRODUCT_BUNDLE_IDENTIFIER = com.local.IntentResourceDemo;" in project_text, "bundle id not set")
    require("CODE_SIGN_STYLE = Automatic;" in project_text, "automatic signing not enabled")
    require("IPHONEOS_DEPLOYMENT_TARGET = 16.0;" in project_text, "deployment target not set")
    build_numbers = re.findall(r"CURRENT_PROJECT_VERSION = ([^;]+);", project_text)
    marketing_versions = re.findall(r"MARKETING_VERSION = ([^;]+);", project_text)
    require(len(build_numbers) == 2, "expected Debug and Release build numbers")
    require(len(set(build_numbers)) == 1, "Debug and Release build numbers differ")
    require(len(marketing_versions) == 2, "expected Debug and Release versions")
    require(len(set(marketing_versions)) == 1, "Debug and Release versions differ")


def validate_translation_flow():
    content_view = CONTENT_VIEW.read_text(encoding="utf-8")
    view_model = DEMO_VIEW_MODEL.read_text(encoding="utf-8")

    require("LanguageAvailability().status(" in content_view, "translation availability is not checked")
    require('static let availabilitySource = Locale.Language(identifier: "zh-Hans")' in content_view, "availability source language not set")
    require('Locale.Language(identifier: "en-US")' in content_view, "translation target language not set")
    require("TranslationSession.Configuration(\n                        source: nil," in content_view, "translation session should detect its source language")
    require("source: SemanticTranslationLanguages.availabilitySource" not in content_view, "translation session source should not be hard-coded")
    require("session.translate(request.sourceText)" in content_view, "translation request is not executed")
    require("try await session.prepareTranslation()" not in content_view, "translation task should call translate directly")
    require('stage: "停止：系统翻译返回空响应"' in content_view, "empty translation response is not diagnosed")
    require("return slots.resourcePhrase" in view_model, "semantic translation does not preserve the resource phrase")
    require("[slots.searchKeyword, slots.resourcePhrase]" not in view_model, "semantic translation source duplicates the query")


def validate_github_actions():
    workflow_text = GITHUB_WORKFLOW.read_text(encoding="utf-8")
    script_text = CI_BUILD_SCRIPT.read_text(encoding="utf-8")
    scheme_text = SCHEME_FILE.read_text(encoding="utf-8")

    require("runs-on: macos-15" in workflow_text, "workflow should use macOS runner")
    require("scripts/ci/build_unsigned_ipa.sh" in workflow_text, "workflow should call IPA build script")
    require("actions/upload-artifact@v4" in workflow_text, "workflow should upload IPA artifact")
    require("CODE_SIGNING_ALLOWED=NO" in script_text, "unsigned build should disable code signing")
    require("IntentResourceDemo-unsigned.ipa" in script_text, "build script should produce unsigned IPA")
    require("BlueprintName = \"IntentResourceDemo\"" in scheme_text, "shared scheme should reference app target")


def main():
    validate_files()
    validate_model()
    validate_resource_index()
    validate_info_plist()
    validate_project_references()
    validate_translation_flow()
    validate_github_actions()
    print("iOS project validation passed")
    print(f"Swift files: {len(SWIFT_FILES)}")
    print(f"Resource files: {len(RESOURCE_FILES)}")
    print(f"Project: {PROJECT_FILE}")


if __name__ == "__main__":
    main()
