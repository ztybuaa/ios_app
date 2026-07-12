import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
from pathlib import Path

from env_guard import ensure_project_venv


ensure_project_venv()

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "external_models" / "pretrained" / "chinese_clip_rn50" / "model_manifest.json"
CHUNK_BYTES = 1024 * 1024
MINIMUM_PARITY_COSINE = 0.999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str) -> None:
    if not path.is_file():
        raise SystemExit(f"Required file is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def package_files(package: Path) -> list[dict[str, object]]:
    result = []
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        result.append(
            {
                "path": path.relative_to(package).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return result


def cosine_similarity(left, right) -> float:
    import numpy as np

    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0 or not math.isfinite(float(denominator)):
        raise SystemExit("Cannot compare an empty or non-finite embedding")
    return float(np.dot(left, right) / denominator)


def validate_contract(
    coreml_model,
    input_name: str,
    input_shape: list[int],
    output_name: str,
    output_size: int,
) -> None:
    spec = coreml_model.get_spec()
    inputs = {feature.name: feature for feature in spec.description.input}
    outputs = {feature.name: feature for feature in spec.description.output}
    if input_name not in inputs:
        raise SystemExit(f"Converted model is missing input {input_name}")
    actual_input_shape = list(inputs[input_name].type.multiArrayType.shape)
    if actual_input_shape != input_shape:
        raise SystemExit(
            f"Converted input {input_name} has shape {actual_input_shape}, expected {input_shape}"
        )
    if output_name not in outputs:
        raise SystemExit(f"Converted model is missing output {output_name}")
    actual_output_shape = list(outputs[output_name].type.multiArrayType.shape)
    actual_output_size = math.prod(actual_output_shape)
    if actual_output_size != output_size:
        raise SystemExit(
            f"Converted output {output_name} has shape {actual_output_shape}, "
            f"expected {output_size} values"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the verified Chinese-CLIP RN50 checkpoint into two FP16 Core ML packages."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing converted output directory after verifying it is the configured directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint = manifest["checkpoint"]
    source = manifest["source"]
    tokenizer = manifest["tokenizer"]
    contract = manifest["coreML"]

    checkpoint_path = ROOT / checkpoint["path"]
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint is missing: {checkpoint_path}. Run the download script first.")
    if checkpoint_path.stat().st_size != checkpoint["bytes"]:
        raise SystemExit(
            f"Checkpoint size mismatch: expected {checkpoint['bytes']}, got {checkpoint_path.stat().st_size}"
        )
    require_sha256(checkpoint_path, checkpoint["sha256"])

    source_root = ROOT / source["path"]
    for relative_path, expected_sha256 in source["verifiedFiles"].items():
        require_sha256(source_root / relative_path, expected_sha256)
    sys.path.insert(0, str(source_root))

    import coremltools as ct
    import numpy as np
    import torch
    from torch import nn

    import cn_clip.clip as clip
    from cn_clip.clip.utils import _MODEL_INFO, create_model

    class TextEncoder(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model

        def forward(self, text):
            return self.clip_model.encode_text(text)

    class ImageEncoder(nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.clip_model = clip_model

        def forward(self, image):
            return self.clip_model.encode_image(image)

    output_root = (ROOT / contract["outputDirectory"]).resolve()
    configured_parent = (ROOT / "external_models" / "converted_ios").resolve()
    if not output_root.is_relative_to(configured_parent):
        raise SystemExit(f"Refusing to write outside {configured_parent}: {output_root}")
    if output_root.exists():
        if not args.force:
            raise SystemExit(f"Output directory already exists: {output_root}. Pass --force to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    print(f"loading verified checkpoint {checkpoint_path}")
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(_MODEL_INFO["RN50"]["struct"], checkpoint_payload).float().eval()

    text_input = clip.tokenize(
        ["小猫图片"], context_length=tokenizer["contextLength"]
    ).int()
    image_input = torch.linspace(
        -1.0,
        1.0,
        steps=3 * 224 * 224,
        dtype=torch.float32,
    ).reshape(1, 3, 224, 224)

    text_encoder = TextEncoder(model).eval()
    image_encoder = ImageEncoder(model).eval()
    with torch.inference_mode():
        expected_text = text_encoder(text_input).detach().cpu().numpy()
        expected_image = image_encoder(image_input).detach().cpu().numpy()

    traced_text = torch.jit.trace(text_encoder, text_input, strict=True)
    traced_image = torch.jit.trace(image_encoder, image_input, strict=True)

    text_coreml = ct.convert(
        traced_text,
        inputs=[
            ct.TensorType(
                name=contract["textInput"]["name"],
                shape=tuple(contract["textInput"]["shape"]),
                dtype=np.int32,
            )
        ],
        outputs=[ct.TensorType(name=contract["textOutput"]["name"])],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )
    image_coreml = ct.convert(
        traced_image,
        inputs=[
            ct.TensorType(
                name=contract["imageInput"]["name"],
                shape=tuple(contract["imageInput"]["shape"]),
                dtype=np.float32,
            )
        ],
        outputs=[ct.TensorType(name=contract["imageOutput"]["name"])],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS15,
    )

    validate_contract(
        text_coreml,
        contract["textInput"]["name"],
        contract["textInput"]["shape"],
        contract["textOutput"]["name"],
        contract["textOutput"]["dimensions"],
    )
    validate_contract(
        image_coreml,
        contract["imageInput"]["name"],
        contract["imageInput"]["shape"],
        contract["imageOutput"]["name"],
        contract["imageOutput"]["dimensions"],
    )

    metadata = {
        "model": manifest["model"]["id"],
        "checkpointRevision": checkpoint["revision"],
        "checkpointSHA256": checkpoint["sha256"],
        "sourceRevision": source["revision"],
        "precision": contract["precision"],
    }
    for key, value in metadata.items():
        text_coreml.user_defined_metadata[key] = value
        image_coreml.user_defined_metadata[key] = value

    text_package = output_root / contract["textModel"]
    image_package = output_root / contract["imageModel"]
    text_coreml.save(text_package)
    image_coreml.save(image_package)

    parity = {"available": platform.system() == "Darwin"}
    if parity["available"]:
        converted_text = text_coreml.predict(
            {contract["textInput"]["name"]: text_input.numpy().astype(np.int32)}
        )[contract["textOutput"]["name"]]
        converted_image = image_coreml.predict(
            {contract["imageInput"]["name"]: image_input.numpy().astype(np.float32)}
        )[contract["imageOutput"]["name"]]
        text_cosine = cosine_similarity(expected_text, converted_text)
        image_cosine = cosine_similarity(expected_image, converted_image)
        parity.update(
            {
                "minimumCosine": MINIMUM_PARITY_COSINE,
                "textCosine": text_cosine,
                "imageCosine": image_cosine,
            }
        )
        if text_cosine < MINIMUM_PARITY_COSINE or image_cosine < MINIMUM_PARITY_COSINE:
            raise SystemExit(
                f"Core ML parity failed: text cosine={text_cosine:.6f}, "
                f"image cosine={image_cosine:.6f}"
            )

    report = {
        "schemaVersion": 1,
        "model": manifest["model"],
        "checkpoint": {
            "revision": checkpoint["revision"],
            "bytes": checkpoint["bytes"],
            "sha256": checkpoint["sha256"],
        },
        "sourceRevision": source["revision"],
        "toolchain": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "numpy": np.__version__,
        },
        "contract": contract,
        "parity": parity,
        "packages": {
            contract["textModel"]: package_files(text_package),
            contract["imageModel"]: package_files(image_package),
        },
    }
    report_path = output_root / "conversion_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"converted text package: {text_package}")
    print(f"converted image package: {image_package}")
    print(f"conversion report: {report_path}")


if __name__ == "__main__":
    main()
