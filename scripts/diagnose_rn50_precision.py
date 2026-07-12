import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image

from env_guard import ensure_project_venv
from eval_chinese_clip_rn50 import (
    MANIFEST_PATH,
    MODEL_MANIFEST_PATH,
    ROOT,
    load_cn_clip,
    normalized,
    validate_checkpoint,
)


ensure_project_venv()

CIFAR_PATH = ROOT / "build" / "diagnostics" / "cifar100" / "test.parquet"
CIFAR_URL = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet"
CIFAR_BYTES = 23_772_751
CIFAR_SHA256 = "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68"
CATS_PATH = ROOT / "processed" / "eval" / "semantic_image_retrieval" / "images" / "cats.jpg"
COCO_ARCHIVE_PATH = ROOT / "build" / "diagnostics" / "coco128" / "coco128.zip"
COCO_ARCHIVE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_ARCHIVE_BYTES = 6_983_030
COCO_ARCHIVE_SHA256 = "61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e"
COCO_IMAGE_ROOT = ROOT / "build" / "diagnostics" / "coco128" / "selected"
COCO_IMAGES = {
    "000000000443": ("positive", 37_330, "f6483d57885c348a27dcdc272e4bd5296c75596d7c64fe638b24dbb503209902"),
    "000000000575": ("positive", 163_333, "839789fe6192bbd5776761d675aa76017b67810b556f3e25e2f3a31acffd86f3"),
    "000000000599": ("positive", 57_584, "9d924b4353cca7295c149e3659866a18e78234f0dc701df119191152f4a31fec"),
    "000000000650": ("positive", 33_865, "9bb45795336c1d05189fe51a77e04270cd1da8929d62060926798f3dd2d25204"),
    "000000000307": ("negative", 96_160, "c839076b4fa14020f97c8a5909c84a1e85c912ca262a91c8c4648ebe27caf61f"),
    "000000000581": ("negative", 45_065, "60ad7a794ae3032c302c4e1d5111099921ce7e8df23df4401f60bdc74dc83bf1"),
}
RESULT_PATH = ROOT / "processed" / "eval" / "semantic_image_retrieval" / "results" / "rn50_cat_precision_stress.json"
SAMPLES_PER_CLASS = 10
NEGATIVE_CLASSES = {
    "baby",
    "bed",
    "bicycle",
    "boy",
    "bus",
    "chair",
    "fox",
    "girl",
    "keyboard",
    "leopard",
    "lion",
    "man",
    "motorcycle",
    "television",
    "tiger",
    "woman",
    "wolf",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cifar100() -> None:
    if CIFAR_PATH.is_file() and CIFAR_PATH.stat().st_size == CIFAR_BYTES:
        if file_sha256(CIFAR_PATH) == CIFAR_SHA256:
            return
    CIFAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial_path = CIFAR_PATH.with_suffix(".part")
    if partial_path.exists():
        partial_path.unlink()
    print(f"Downloading pinned CIFAR-100 stress set: {CIFAR_URL}")
    urllib.request.urlretrieve(CIFAR_URL, partial_path)
    actual_bytes = partial_path.stat().st_size
    actual_sha256 = file_sha256(partial_path)
    if actual_bytes != CIFAR_BYTES or actual_sha256 != CIFAR_SHA256:
        partial_path.unlink()
        raise SystemExit(
            "CIFAR-100 stress set integrity mismatch: "
            f"bytes={actual_bytes} sha256={actual_sha256}"
        )
    partial_path.replace(CIFAR_PATH)


def ensure_coco_archive() -> None:
    if not (
        COCO_ARCHIVE_PATH.is_file()
        and COCO_ARCHIVE_PATH.stat().st_size == COCO_ARCHIVE_BYTES
        and file_sha256(COCO_ARCHIVE_PATH) == COCO_ARCHIVE_SHA256
    ):
        COCO_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial_path = COCO_ARCHIVE_PATH.with_suffix(".part")
        if partial_path.exists():
            partial_path.unlink()
        print(f"Downloading pinned COCO128 stress set: {COCO_ARCHIVE_URL}")
        urllib.request.urlretrieve(COCO_ARCHIVE_URL, partial_path)
        actual_bytes = partial_path.stat().st_size
        actual_sha256 = file_sha256(partial_path)
        if actual_bytes != COCO_ARCHIVE_BYTES or actual_sha256 != COCO_ARCHIVE_SHA256:
            partial_path.unlink()
            raise SystemExit(
                "COCO128 stress set integrity mismatch: "
                f"bytes={actual_bytes} sha256={actual_sha256}"
            )
        partial_path.replace(COCO_ARCHIVE_PATH)


def ensure_coco_images() -> None:
    expected_images_ready = all(
        (COCO_IMAGE_ROOT / f"{image_id}.jpg").is_file()
        and (COCO_IMAGE_ROOT / f"{image_id}.jpg").stat().st_size == expected_bytes
        and file_sha256(COCO_IMAGE_ROOT / f"{image_id}.jpg") == expected_sha256
        for image_id, (_, expected_bytes, expected_sha256) in COCO_IMAGES.items()
    )
    if expected_images_ready:
        return

    ensure_coco_archive()

    COCO_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(COCO_ARCHIVE_PATH) as archive:
        for image_id, (_, expected_bytes, expected_sha256) in COCO_IMAGES.items():
            data = archive.read(f"coco128/images/train2017/{image_id}.jpg")
            actual_sha256 = hashlib.sha256(data).hexdigest()
            if len(data) != expected_bytes or actual_sha256 != expected_sha256:
                raise SystemExit(
                    f"COCO image integrity mismatch for {image_id}: "
                    f"bytes={len(data)} sha256={actual_sha256}"
                )
            (COCO_IMAGE_ROOT / f"{image_id}.jpg").write_bytes(data)


def load_stress_images() -> list[tuple[str, Image.Image]]:
    ensure_cifar100()
    ensure_coco_images()

    table = pq.read_table(CIFAR_PATH)
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    label_names = metadata["info"]["features"]["fine_label"]["names"]
    selected_counts = {name: 0 for name in NEGATIVE_CLASSES}
    images: list[tuple[str, Image.Image]] = []

    with Image.open(CATS_PATH) as cats:
        images.append(("positive-cats/full", cats.convert("RGB").copy()))

    for image_id, (kind, _, _) in COCO_IMAGES.items():
        with Image.open(COCO_IMAGE_ROOT / f"{image_id}.jpg") as image:
            images.append((f"{kind}-coco-{image_id}", image.convert("RGB").copy()))

    for row in table.to_pylist():
        label = label_names[row["fine_label"]]
        if label not in selected_counts or selected_counts[label] >= SAMPLES_PER_CLASS:
            continue
        index = selected_counts[label]
        selected_counts[label] += 1
        images.append((f"negative-{label}-{index:02d}", Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")))

    missing = [name for name, count in selected_counts.items() if count != SAMPLES_PER_CLASS]
    if missing:
        raise SystemExit(f"CIFAR-100 stress classes are incomplete: {missing}")
    return images


def main() -> None:
    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint_path, _ = validate_checkpoint(model_manifest)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    query = next(item for item in manifest["queries"] if item["id"] == "cat")
    clip = load_cn_clip(ROOT / model_manifest["source"]["path"])

    from cn_clip.clip.utils import _MODEL_INFO, create_model, image_transform

    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(_MODEL_INFO["RN50"]["struct"], checkpoint_payload).float().eval()
    preprocess = image_transform(224)
    images = load_stress_images()

    started_at = time.perf_counter()
    with torch.inference_mode():
        image_features = normalized(model.encode_image(torch.stack([preprocess(image) for _, image in images])))
        text_inputs = clip.tokenize([query["chinese"]] + query["hardNegatives"])
        text_features = normalized(model.encode_text(text_inputs))
    inference_ms = (time.perf_counter() - started_at) * 1_000

    positive_scores = image_features @ text_features[0]
    negative_scores = (image_features @ text_features[1:].T).max(dim=1).values
    margins = positive_scores - negative_scores
    minimum_similarity = float(manifest["calibration"]["minimumSimilarity"])
    minimum_margin = float(query["minimumMargin"])

    passing = []
    for index, (image_id, _) in enumerate(images):
        positive = float(positive_scores[index].item())
        margin = float(margins[index].item())
        if positive >= minimum_similarity and margin >= minimum_margin:
            passing.append((image_id, positive, margin))

    positive_passes = [item for item in passing if item[0].startswith("positive-")]
    false_positives = [item for item in passing if item[0].startswith("negative-")]
    print(
        f"stressImages={len(images)} inferenceMs={inference_ms:.2f} "
        f"positiveImages={len(positive_passes)} falsePositives={len(false_positives)}"
    )
    for image_id, positive, margin in sorted(false_positives, key=lambda item: item[2], reverse=True):
        print(f"FALSE_POSITIVE {image_id} similarity={positive:.4f} margin={margin:.4f}")

    expected_positive_count = 1 + sum(1 for kind, _, _ in COCO_IMAGES.values() if kind == "positive")
    passed = len(positive_passes) == expected_positive_count and not false_positives
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(
            {
                "version": 1,
                "query": query["chinese"],
                "hardNegatives": query["hardNegatives"],
                "dataset": {
                    "cifar100": {
                        "url": CIFAR_URL,
                        "bytes": CIFAR_BYTES,
                        "sha256": CIFAR_SHA256,
                        "license": "Not specified by the source dataset card; used only for local evaluation.",
                    },
                    "coco128": {
                        "url": COCO_ARCHIVE_URL,
                        "bytes": COCO_ARCHIVE_BYTES,
                        "sha256": COCO_ARCHIVE_SHA256,
                        "license": "Ultralytics archive metadata is GPL-3.0; COCO images retain their source licenses.",
                    },
                    "sampleCount": len(images),
                    "samplesPerNegativeClass": SAMPLES_PER_CLASS,
                    "negativeClasses": sorted(NEGATIVE_CLASSES),
                },
                "minimumSimilarity": minimum_similarity,
                "minimumMargin": minimum_margin,
                "expectedPositiveCount": expected_positive_count,
                "positivePasses": [image_id for image_id, _, _ in positive_passes],
                "falsePositives": [
                    {"id": image_id, "similarity": positive, "margin": margin}
                    for image_id, positive, margin in false_positives
                ],
                "inferenceMs": inference_ms,
                "passed": passed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if len(positive_passes) != expected_positive_count:
        raise SystemExit(
            f"FAIL: expected {expected_positive_count} cat images, "
            f"but only {len(positive_passes)} passed"
        )
    if false_positives:
        raise SystemExit(f"FAIL: {len(false_positives)} non-cat images passed the cat filter")
    print("PASS: cat recall retained with zero stress-set false positives")


if __name__ == "__main__":
    main()
