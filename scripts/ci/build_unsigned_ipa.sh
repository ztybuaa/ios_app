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

echo "Unsigned IPA generated: $IPA_PATH"
