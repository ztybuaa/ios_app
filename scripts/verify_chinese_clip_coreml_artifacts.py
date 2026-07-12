import argparse
import hashlib
import json
from pathlib import Path

from env_guard import ensure_project_venv


ensure_project_venv()

CHUNK_BYTES = 1024 * 1024
MINIMUM_PARITY_COSINE = 0.999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_json(path: Path) -> dict:
    require(path.is_file(), f"Required manifest is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot parse manifest {path}: {error}") from error


def verify_package(package: Path, entries: list[dict]) -> None:
    require(package.is_dir(), f"Core ML package is missing: {package}")
    expected_paths: set[str] = set()

    for entry in entries:
        require(isinstance(entry, dict), "Invalid package entry in conversion manifest")
        relative_path = entry.get("path")
        require(
            isinstance(relative_path, str) and relative_path,
            "Invalid package path in conversion manifest",
        )
        require(relative_path not in expected_paths, f"Duplicate package path: {relative_path}")
        expected_paths.add(relative_path)

        path = (package / relative_path).resolve()
        require(path.is_relative_to(package.resolve()), f"Package path escapes its root: {relative_path}")
        require(path.is_file(), f"Converted package file is missing: {path}")
        expected_bytes = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        require(isinstance(expected_bytes, int) and expected_bytes >= 0, f"Invalid byte count: {relative_path}")
        require(
            isinstance(expected_sha256, str) and len(expected_sha256) == 64,
            f"Invalid SHA-256: {relative_path}",
        )
        require(path.stat().st_size == expected_bytes, f"Converted package size mismatch: {path}")
        require(sha256(path) == expected_sha256, f"Converted package SHA-256 mismatch: {path}")

    actual_paths = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
    }
    require(actual_paths == expected_paths, f"Converted package file set differs from its manifest: {package}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify converted Chinese-CLIP Core ML package provenance and bytes.")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_manifest = load_json(args.model_manifest.resolve())
    conversion_manifest_path = args.conversion_manifest.resolve()
    conversion_manifest = load_json(conversion_manifest_path)

    require(
        conversion_manifest.get("model") == model_manifest.get("model", {}).get("id"),
        "Converted model ID does not match the pinned model manifest",
    )
    require(
        conversion_manifest.get("checkpoint", {}).get("revision")
        == model_manifest.get("checkpoint", {}).get("revision"),
        "Converted checkpoint revision does not match the pinned model manifest",
    )
    require(
        conversion_manifest.get("checkpoint", {}).get("sha256")
        == model_manifest.get("checkpoint", {}).get("sha256"),
        "Converted checkpoint SHA-256 does not match the pinned model manifest",
    )
    require(
        conversion_manifest.get("sourceRevision") == model_manifest.get("source", {}).get("revision"),
        "Converted source revision does not match the pinned model manifest",
    )
    require(
        conversion_manifest.get("contract", {}).get("precision")
        == model_manifest.get("coreML", {}).get("precision"),
        "Converted precision does not match the pinned model manifest",
    )

    parity = conversion_manifest.get("parity", {})
    require(parity.get("available") is True, "Core ML parity was not measured on macOS")
    for tower in ("textCosine", "imageCosine"):
        value = parity.get(tower)
        require(isinstance(value, (int, float)), f"Core ML parity value is missing: {tower}")
        require(value >= MINIMUM_PARITY_COSINE, f"Core ML parity is below {MINIMUM_PARITY_COSINE}: {tower}={value}")

    contract = model_manifest["coreML"]
    expected_packages = {contract["textModel"], contract["imageModel"]}
    packages = conversion_manifest.get("packages", {})
    require(set(packages) == expected_packages, "Conversion manifest contains an unexpected package set")

    output_directory = conversion_manifest_path.parent
    for package_name in sorted(expected_packages):
        entries = packages[package_name]
        require(isinstance(entries, list) and entries, f"No files recorded for package: {package_name}")
        verify_package(output_directory / package_name, entries)

    print("Chinese-CLIP Core ML artifact verification passed")


if __name__ == "__main__":
    main()
