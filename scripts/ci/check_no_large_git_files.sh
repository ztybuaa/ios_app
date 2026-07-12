#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MAX_GIT_BLOB_BYTES=$((100 * 1024 * 1024))

cd "$ROOT_DIR"

largest_size=0
largest_path=""
failed=0

while IFS= read -r -d '' path; do
  if [ ! -f "$path" ]; then
    continue
  fi

  if size=$(stat -f '%z' "$path" 2>/dev/null); then
    :
  else
    size=$(stat -c '%s' "$path")
  fi

  if [ "$size" -gt "$largest_size" ]; then
    largest_size="$size"
    largest_path="$path"
  fi

  if [ "$size" -ge "$MAX_GIT_BLOB_BYTES" ]; then
    echo "Tracked file is at or above the 100 MiB Git limit: $path ($size bytes)" >&2
    failed=1
  fi
done < <(git ls-files -z)

if [ "$failed" -ne 0 ]; then
  echo "Store model checkpoints and converted Core ML packages outside ordinary Git." >&2
  exit 1
fi

echo "Git file-size check passed. Largest tracked file: $largest_path ($largest_size bytes)"
