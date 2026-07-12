#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CONVERTED_DIR="$ROOT_DIR/external_models/converted_ios/chinese_clip_rn50_fp16"
APP_MODEL_DIR="$ROOT_DIR/ios_app/IntentResourceDemo/Resources/ChineseCLIP"
MODEL_MANIFEST="$ROOT_DIR/external_models/pretrained/chinese_clip_rn50/model_manifest.json"
CONVERSION_MANIFEST="$CONVERTED_DIR/conversion_manifest.json"

MODEL_NAMES=(
  "chinese_clip_rn50_image.mlpackage"
  "chinese_clip_rn50_text.mlpackage"
)

if [ ! -f "$MODEL_MANIFEST" ] || [ ! -f "$CONVERSION_MANIFEST" ]; then
  echo "Model or conversion manifest is missing." >&2
  exit 1
fi

EXPECTED_CHECKPOINT_SHA="$(/usr/bin/plutil -extract checkpoint.sha256 raw -o - "$MODEL_MANIFEST")"
EXPECTED_SOURCE_REVISION="$(/usr/bin/plutil -extract source.revision raw -o - "$MODEL_MANIFEST")"
ACTUAL_CHECKPOINT_SHA="$(/usr/bin/plutil -extract checkpoint.sha256 raw -o - "$CONVERSION_MANIFEST")"
ACTUAL_SOURCE_REVISION="$(/usr/bin/plutil -extract sourceRevision raw -o - "$CONVERSION_MANIFEST")"
PARITY_AVAILABLE="$(/usr/bin/plutil -extract parity.available raw -o - "$CONVERSION_MANIFEST")"
TEXT_COSINE="$(/usr/bin/plutil -extract parity.textCosine raw -o - "$CONVERSION_MANIFEST")"
IMAGE_COSINE="$(/usr/bin/plutil -extract parity.imageCosine raw -o - "$CONVERSION_MANIFEST")"

if [ "$ACTUAL_CHECKPOINT_SHA" != "$EXPECTED_CHECKPOINT_SHA" ] || \
   [ "$ACTUAL_SOURCE_REVISION" != "$EXPECTED_SOURCE_REVISION" ]; then
  echo "Converted Core ML provenance does not match the pinned model manifest." >&2
  exit 1
fi
if [ "$PARITY_AVAILABLE" != "true" ] || \
   ! awk -v value="$TEXT_COSINE" 'BEGIN { exit !(value >= 0.999) }' || \
   ! awk -v value="$IMAGE_COSINE" 'BEGIN { exit !(value >= 0.999) }'; then
  echo "Converted Core ML parity is missing or below 0.999." >&2
  exit 1
fi

for model_name in "${MODEL_NAMES[@]}"; do
  source_path="$CONVERTED_DIR/$model_name"
  if [ ! -f "$source_path/Manifest.json" ]; then
    echo "Converted Core ML package is missing its manifest: $source_path" >&2
    exit 1
  fi
  if [ ! -f "$source_path/Data/com.apple.CoreML/model.mlmodel" ]; then
    echo "Converted Core ML package is missing model.mlmodel: $source_path" >&2
    exit 1
  fi
  if ! find "$source_path/Data/com.apple.CoreML/weights" -type f -size +0c -print -quit | grep -q .; then
    echo "Converted Core ML package has no non-empty weights: $source_path" >&2
    exit 1
  fi
done

mkdir -p "$APP_MODEL_DIR"
for model_name in "${MODEL_NAMES[@]}"; do
  /usr/bin/ditto \
    "$CONVERTED_DIR/$model_name" \
    "$APP_MODEL_DIR/$model_name"
done

echo "Chinese-CLIP Core ML packages staged in: $APP_MODEL_DIR"
