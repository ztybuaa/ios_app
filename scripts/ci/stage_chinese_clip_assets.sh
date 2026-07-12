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

"$ROOT_DIR/.venv/bin/python" \
  "$ROOT_DIR/scripts/verify_chinese_clip_coreml_artifacts.py" \
  --model-manifest "$MODEL_MANIFEST" \
  --conversion-manifest "$CONVERSION_MANIFEST"

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
