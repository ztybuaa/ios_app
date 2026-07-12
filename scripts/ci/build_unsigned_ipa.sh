#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PROJECT_PATH="$ROOT_DIR/ios_app/IntentResourceDemo.xcodeproj"
SCHEME="IntentResourceDemo"
CONFIGURATION="Release"
DERIVED_DATA="$ROOT_DIR/build/DerivedData"
OUTPUT_DIR="$ROOT_DIR/build/ipa"
APP_PATH="$DERIVED_DATA/Build/Products/${CONFIGURATION}-iphoneos/IntentResourceDemo.app"
PAYLOAD_DIR="$OUTPUT_DIR/Payload"
IPA_PATH="$OUTPUT_DIR/IntentResourceDemo-unsigned.ipa"
CHECKSUM_PATH="$IPA_PATH.sha256"
BUILD_INFO_PATH="$OUTPUT_DIR/ipa-build.json"
MODEL_MANIFEST="$ROOT_DIR/external_models/pretrained/chinese_clip_rn50/model_manifest.json"

rm -rf "$DERIVED_DATA" "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

xcodebuild \
  -project "$PROJECT_PATH" \
  -scheme "$SCHEME" \
  -configuration "$CONFIGURATION" \
  -destination "generic/platform=iOS" \
  -derivedDataPath "$DERIVED_DATA" \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY="" \
  clean build

if [ ! -d "$APP_PATH" ]; then
  echo "Expected app bundle not found: $APP_PATH" >&2
  find "$DERIVED_DATA/Build/Products" -maxdepth 3 -type d -name "*.app" -print >&2 || true
  exit 1
fi

mkdir -p "$PAYLOAD_DIR"
cp -R "$APP_PATH" "$PAYLOAD_DIR/"

(
  cd "$OUTPUT_DIR"
  /usr/bin/zip -qry "$IPA_PATH" Payload
)

unzip -qt "$IPA_PATH"

IPA_SHA256="$(shasum -a 256 "$IPA_PATH" | awk '{print $1}')"
IPA_BYTES="$(stat -f '%z' "$IPA_PATH")"
APP_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP_PATH/Info.plist")"
APP_BUILD="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$APP_PATH/Info.plist")"
HEAD_SHA="${GITHUB_SHA:-$(git -C "$ROOT_DIR" rev-parse HEAD)}"
RUN_NUMBER="${GITHUB_RUN_NUMBER:-0}"
MODEL_SOURCE="$(/usr/bin/plutil -extract model.id raw -o - "$MODEL_MANIFEST")"
MODEL_PRECISION="$(/usr/bin/plutil -extract coreML.precision raw -o - "$MODEL_MANIFEST")"
MODEL_CHECKPOINT_REVISION="$(/usr/bin/plutil -extract checkpoint.revision raw -o - "$MODEL_MANIFEST")"
MODEL_CHECKPOINT_SHA256="$(/usr/bin/plutil -extract checkpoint.sha256 raw -o - "$MODEL_MANIFEST")"
MODEL_SOURCE_REVISION="$(/usr/bin/plutil -extract source.revision raw -o - "$MODEL_MANIFEST")"

printf '%s  %s\n' "$IPA_SHA256" "$(basename "$IPA_PATH")" > "$CHECKSUM_PATH"
cat > "$BUILD_INFO_PATH" <<EOF
{
  "artifact": "$(basename "$IPA_PATH")",
  "sha256": "$IPA_SHA256",
  "bytes": $IPA_BYTES,
  "appVersion": "$APP_VERSION",
  "appBuild": "$APP_BUILD",
  "model": "chinese-clip-rn50-fp16",
  "modelSource": "$MODEL_SOURCE",
  "modelPrecision": "$MODEL_PRECISION",
  "modelCheckpointRevision": "$MODEL_CHECKPOINT_REVISION",
  "modelCheckpointSHA256": "$MODEL_CHECKPOINT_SHA256",
  "modelSourceRevision": "$MODEL_SOURCE_REVISION",
  "headSha": "$HEAD_SHA",
  "githubRunNumber": $RUN_NUMBER
}
EOF

echo "Unsigned IPA generated: $IPA_PATH"
echo "IPA SHA-256: $IPA_SHA256"
echo "IPA bytes: $IPA_BYTES"
