import argparse
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
SEMANTIC_SEARCH = APP_ROOT / "ResourceModules" / "SemanticImageSearchService.swift"
MEDIA_RESOURCE = APP_ROOT / "ResourceModules" / "MediaResourceModule.swift"
CHINESE_CLIP_TOKENIZER = APP_ROOT / "NLP" / "Tokenizer" / "ChineseCLIPTokenizer.swift"

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
    "NLP/Tokenizer/ChineseCLIPTokenizer.swift",
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
    "Resources/chinese_clip_vocab.txt",
]

CHINESE_CLIP_FILES = [
    "Resources/ChineseCLIP/chinese_clip_rn50_image.mlpackage/Manifest.json",
    "Resources/ChineseCLIP/chinese_clip_rn50_image.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "Resources/ChineseCLIP/chinese_clip_rn50_image.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "Resources/ChineseCLIP/chinese_clip_rn50_text.mlpackage/Manifest.json",
    "Resources/ChineseCLIP/chinese_clip_rn50_text.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "Resources/ChineseCLIP/chinese_clip_rn50_text.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
]


def require(condition, message):
    if not condition:
        raise SystemExit(f"VALIDATION FAILED: {message}")


def validate_files(require_generated_models):
    require(PROJECT_FILE.exists(), f"missing {PROJECT_FILE}")
    require(SCHEME_FILE.exists(), f"missing shared scheme {SCHEME_FILE}")
    require(GITHUB_WORKFLOW.exists(), f"missing GitHub Actions workflow {GITHUB_WORKFLOW}")
    require(CI_BUILD_SCRIPT.exists(), f"missing CI build script {CI_BUILD_SCRIPT}")
    required_files = SWIFT_FILES + RESOURCE_FILES + ["Support/Info.plist"]
    if require_generated_models:
        required_files += CHINESE_CLIP_FILES
    for relative in required_files:
        require((APP_ROOT / relative).exists(), f"missing app file {relative}")
    for relative in (
        "NLP/Tokenizer/CLIPTokenizer.swift",
        "NLP/Tokenizer/GPT2ByteEncoder.swift",
        "NLP/Tokenizer/Utils.swift",
        "Resources/clip-vocab.json",
        "Resources/clip-merges.txt",
        "Resources/MobileCLIP",
    ):
        require(not (APP_ROOT / relative).exists(), f"obsolete app file still exists: {relative}")


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
    for package_name in ("chinese_clip_rn50_image.mlpackage", "chinese_clip_rn50_text.mlpackage"):
        require(package_name in project_text, f"project.pbxproj missing reference to {package_name}")
        require(f"{package_name} in Sources" in project_text, f"{package_name} should be in Sources for Core ML codegen")
    for obsolete_name in ("MobileCLIP", "mobileclip_s0", "clip-vocab.json", "clip-merges.txt"):
        require(obsolete_name not in project_text, f"project.pbxproj still references obsolete {obsolete_name}")
    require("PRODUCT_BUNDLE_IDENTIFIER = com.local.IntentResourceDemo;" in project_text, "bundle id not set")
    require("CODE_SIGN_STYLE = Automatic;" in project_text, "automatic signing not enabled")
    require("IPHONEOS_DEPLOYMENT_TARGET = 16.0;" in project_text, "deployment target not set")
    build_numbers = re.findall(r"CURRENT_PROJECT_VERSION = ([^;]+);", project_text)
    marketing_versions = re.findall(r"MARKETING_VERSION = ([^;]+);", project_text)
    require(len(build_numbers) == 2, "expected Debug and Release build numbers")
    require(len(set(build_numbers)) == 1, "Debug and Release build numbers differ")
    require(len(marketing_versions) == 2, "expected Debug and Release versions")
    require(len(set(marketing_versions)) == 1, "Debug and Release versions differ")


def validate_chinese_clip_flow():
    content_view = CONTENT_VIEW.read_text(encoding="utf-8")
    view_model = DEMO_VIEW_MODEL.read_text(encoding="utf-8")
    semantic_search = SEMANTIC_SEARCH.read_text(encoding="utf-8")
    media_resource = MEDIA_RESOURCE.read_text(encoding="utf-8")
    tokenizer = CHINESE_CLIP_TOKENIZER.read_text(encoding="utf-8")

    for obsolete_term in ("import Translation", "TranslationSession", "TranslationDiagnosticSection", "semanticQueryText", "pendingTranslation"):
        require(obsolete_term not in content_view + view_model, f"obsolete translation flow still contains {obsolete_term}")
    require('textModelName = "chinese_clip_rn50_text"' in semantic_search, "Chinese-CLIP text model name is missing")
    require('imageModelName = "chinese_clip_rn50_image"' in semantic_search, "Chinese-CLIP image model name is missing")
    require('inputShape: [1, 52]' in semantic_search, "text input must be [1, 52]")
    require('inputShape: [1, 3, 224, 224]' in semantic_search, "image input must be [1, 3, 224, 224]")
    require('output: "text_features"' in semantic_search, "text output contract is missing")
    require('output: "image_features"' in semantic_search, "image output contract is missing")
    require('static let embeddingDimensions = 1_024' in semantic_search, "embedding dimension must be 1024")
    require("positivePrompts: orderedUnique([subject])" in semantic_search, "Chinese prompt planner must use the raw Chinese subject")
    require("static let minimumSimilarity: Float = 0.47" in semantic_search, "Chinese-CLIP similarity threshold is not calibrated")
    require("static let screenshotMinimumMargin: Float = 0.011" in semantic_search, "screenshot margin threshold is not calibrated")
    require("static let defaultMinimumMargin: Float = 0.012" in semantic_search, "default margin threshold is not calibrated")
    require(
        "case .coarse:" in semantic_search
        and "options.deliveryMode = .fastFormat" in semantic_search
        and "case .full:" in semantic_search
        and "options.deliveryMode = .highQualityFormat" in semantic_search,
        "semantic retrieval must separate fast coarse images from high-quality final images",
    )
    require(
        "PHImageResultIsDegradedKey" in semantic_search,
        "semantic retrieval must distinguish degraded Photos callbacks",
    )
    require(
        "preprocess-v2-two-stage-quality-v1" in semantic_search,
        "semantic embedding cache namespace must invalidate low-quality thumbnails",
    )
    require(
        "qualityShortlist.map(\\.asset)" in semantic_search
        and "profile: .coarse" in semantic_search
        and "profile: .full" in semantic_search
        and "let candidates = qualityMatches" in semantic_search,
        "semantic retrieval must rank only candidates recomputed from high-quality images",
    )
    require(
        "verifiedSemanticCandidates" not in media_resource
        and "requiresSemanticVerification" not in media_resource,
        "query-specific Vision verification must not bypass the general semantic pipeline",
    )
    require(
        "PhotoImageRequestState" in semantic_search
        and "withTaskCancellationHandler" in semantic_search
        and "profile.requestTimeout" in semantic_search,
        "PhotoKit requests must finish on cancellation or timeout",
    )
    require(
        "orientation: orientation" in media_resource,
        "Vision analysis must preserve image orientation",
    )
    require("try?" not in semantic_search, "semantic model errors must not be silently discarded")
    require("static let contextLength = 52" in tokenizer, "tokenizer context length must be 52")
    require("static let vocabularySize = 21_128" in tokenizer, "tokenizer vocabulary size must be 21128")
    require('vocabulary["[CLS]"] == 101' in tokenizer, "tokenizer must validate [CLS]")
    require('vocabulary["[SEP]"] == 102' in tokenizer, "tokenizer must validate [SEP]")

    vocab_path = APP_ROOT / "Resources" / "chinese_clip_vocab.txt"
    vocab = vocab_path.read_text(encoding="utf-8").split("\n")
    if vocab and vocab[-1] == "":
        vocab.pop()
    vocab = [line[:-1] if line.endswith("\r") else line for line in vocab]
    require(len(vocab) == 21_128, "Chinese-CLIP vocabulary must contain 21128 tokens")
    for token, expected_id in {
        "[PAD]": 0,
        "[UNK]": 100,
        "[CLS]": 101,
        "[SEP]": 102,
        "小": 2207,
        "猫": 4344,
        "图": 1745,
        "片": 4275,
    }.items():
        require(vocab[expected_id] == token, f"Chinese-CLIP vocabulary token {token} has the wrong ID")
    require(vocab[343] == chr(0x2028), "Chinese-CLIP vocabulary lost the U+2028 token at ID 343")
    require(vocab[13_502] == "##" + chr(0x2028), "Chinese-CLIP vocabulary lost the ##U+2028 token at ID 13502")


def validate_github_actions():
    workflow_text = GITHUB_WORKFLOW.read_text(encoding="utf-8")
    script_text = CI_BUILD_SCRIPT.read_text(encoding="utf-8")
    scheme_text = SCHEME_FILE.read_text(encoding="utf-8")

    require("runs-on: macos-15" in workflow_text, "workflow should use macOS runner")
    require("DEVELOPER_DIR: /Applications/Xcode_16.3.app/Contents/Developer" in workflow_text, "workflow should pin Xcode 16.3 for iOS 18.4 compatibility")
    require("xcrun --sdk iphoneos --show-sdk-version" in workflow_text, "workflow should report the selected iOS SDK")
    require("scripts/ci/build_unsigned_ipa.sh" in workflow_text, "workflow should call IPA build script")
    require("scripts/eval_chinese_clip_rn50.py" in workflow_text, "workflow should run semantic retrieval evaluation")
    require(
        "scripts/validate_chinese_clip_multiclass_quality.py" in workflow_text,
        "workflow should run multiclass quality and shortlist recall gates",
    )
    require(
        "scripts/validate_semantic_search_performance.py" in workflow_text,
        "workflow should enforce the bounded two-stage performance policy",
    )
    require("scripts/diagnose_rn50_precision.py" in workflow_text, "workflow should run semantic precision stress tests")
    require("actions/upload-artifact@v4" in workflow_text, "workflow should upload IPA artifact")
    require("CODE_SIGNING_ALLOWED=NO" in script_text, "unsigned build should disable code signing")
    require("IntentResourceDemo-unsigned.ipa" in script_text, "build script should produce unsigned IPA")
    require("BlueprintName = \"IntentResourceDemo\"" in scheme_text, "shared scheme should reference app target")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-generated-models",
        action="store_true",
        help="Require CI-generated Chinese-CLIP Core ML packages to be staged.",
    )
    args = parser.parse_args()

    validate_files(args.require_generated_models)
    validate_model()
    validate_resource_index()
    validate_info_plist()
    validate_project_references()
    validate_chinese_clip_flow()
    validate_github_actions()
    print("iOS project validation passed")
    print(f"Swift files: {len(SWIFT_FILES)}")
    print(f"Resource files: {len(RESOURCE_FILES)}")
    print(f"Project: {PROJECT_FILE}")


if __name__ == "__main__":
    main()
