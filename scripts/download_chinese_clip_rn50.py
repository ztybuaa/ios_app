import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from env_guard import ensure_project_venv


ensure_project_venv()

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "external_models" / "pretrained" / "chinese_clip_rn50" / "model_manifest.json"
SOURCE_ARCHIVE_ROOT = ROOT / "build" / "external_downloads"
CHUNK_BYTES = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Required file is missing: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"Unexpected file size for {path}: expected {expected_bytes}, got {actual_bytes}. "
            "Delete the invalid file and run the downloader again."
        )
    actual_sha256 = sha256(path)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}. "
            "Delete the invalid file and run the downloader again."
        )


def download_verified(url: str, destination: Path, expected_bytes: int, expected_sha256: str) -> None:
    if destination.exists():
        verify_file(destination, expected_bytes, expected_sha256)
        print(f"verified existing {destination.relative_to(ROOT)}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        raise SystemExit(
            f"Partial download is larger than expected: {partial}. Delete it before retrying."
        )

    headers = {"User-Agent": "IntentResourceDemo-model-fetch/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"resuming {destination.name} at byte {offset:,}")
    else:
        print(f"downloading {destination.name} ({expected_bytes:,} bytes)")

    request = urllib.request.Request(url, headers=headers)
    started_at = time.monotonic()
    last_report = started_at
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if offset and status != 206:
            raise SystemExit(
                f"Server ignored the Range request for {destination.name}: HTTP {status}. "
                "The partial file was kept; retry against a server with Range support."
            )
        if not offset and status != 200:
            raise SystemExit(f"Unexpected HTTP {status} while downloading {destination.name}")

        mode = "ab" if offset else "wb"
        downloaded = offset
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                stream.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    percent = downloaded * 100 / expected_bytes
                    rate = (downloaded - offset) / max(now - started_at, 0.001) / (1024 * 1024)
                    print(f"  {downloaded:,}/{expected_bytes:,} bytes ({percent:.1f}%, {rate:.2f} MiB/s)")
                    last_report = now

    verify_file(partial, expected_bytes, expected_sha256)
    os.replace(partial, destination)
    print(f"verified download {destination.relative_to(ROOT)} sha256={expected_sha256}")


def verify_source(source_dir: Path, verified_files: dict[str, str]) -> None:
    if not source_dir.is_dir():
        raise SystemExit(f"Chinese-CLIP source directory is missing: {source_dir}")
    for relative_path, expected_sha256 in verified_files.items():
        path = source_dir / relative_path
        if not path.is_file():
            raise SystemExit(f"Chinese-CLIP source file is missing: {path}")
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"Chinese-CLIP source hash mismatch for {relative_path}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )


def extract_source(archive: Path, source_dir: Path, revision: str, verified_files: dict[str, str]) -> None:
    if source_dir.exists():
        verify_source(source_dir, verified_files)
        print(f"verified existing {source_dir.relative_to(ROOT)}")
        return

    source_dir.parent.mkdir(parents=True, exist_ok=True)
    expected_root = f"Chinese-CLIP-{revision}"
    with tempfile.TemporaryDirectory(prefix="chinese_clip_source_", dir=source_dir.parent) as temporary:
        temporary_root = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                target = (temporary_root / member.filename).resolve()
                if not target.is_relative_to(temporary_root):
                    raise SystemExit(f"Unsafe path in Chinese-CLIP source archive: {member.filename}")
            package.extractall(temporary_root)

        extracted_root = temporary_root / expected_root
        verify_source(extracted_root, verified_files)
        shutil.move(str(extracted_root), str(source_dir))

    verify_source(source_dir, verified_files)
    print(f"verified source {source_dir.relative_to(ROOT)} revision={revision}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and strictly verify the official Chinese-CLIP RN50 checkpoint and source."
    )
    parser.add_argument(
        "--skip-source",
        action="store_true",
        help="Only verify/download the checkpoint. Core ML conversion requires the source checkout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    checkpoint = manifest["checkpoint"]
    checkpoint_path = ROOT / checkpoint["path"]
    download_verified(
        checkpoint["url"],
        checkpoint_path,
        checkpoint["bytes"],
        checkpoint["sha256"],
    )

    if args.skip_source:
        return

    source = manifest["source"]
    source_archive = SOURCE_ARCHIVE_ROOT / f"Chinese-CLIP-{source['revision']}.zip"
    download_verified(
        source["archiveUrl"],
        source_archive,
        source["archiveBytes"],
        source["archiveSha256"],
    )
    extract_source(
        source_archive,
        ROOT / source["path"],
        source["revision"],
        source["verifiedFiles"],
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("download interrupted; the .part file is retained for the next run", file=sys.stderr)
        raise SystemExit(130)
