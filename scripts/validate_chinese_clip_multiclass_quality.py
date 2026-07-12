import io
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from diagnose_rn50_precision import COCO_ARCHIVE_PATH, ensure_coco_archive
from env_guard import ensure_project_venv
from eval_chinese_clip_rn50 import (
    MODEL_MANIFEST_PATH,
    ROOT,
    load_cn_clip,
    normalized,
    validate_checkpoint,
)


ensure_project_venv()

COCO_IMAGE_PREFIX = "coco128/images/train2017/"
COCO_LABEL_PREFIX = "coco128/labels/train2017/"
EXPECTED_IMAGE_COUNT = 128
EXPECTED_EMPTY_LABEL_COUNT = 2
COARSE_SIDE = 256
ORACLE_TOP_K = 12
SHORTLIST_K = 64
MINIMUM_TOP1_HITS = 7
MINIMUM_MACRO_PRECISION_AT_5 = 0.45
BATCH_SIZE = 32


@dataclass(frozen=True)
class QuerySpec:
    name: str
    class_id: int
    positive: str
    negatives: tuple[str, ...]


GENERIC_NEGATIVES = ("与查询内容无关的图片", "随机图片", "模糊图片")
QUERIES = (
    QuerySpec("person", 0, "人物图片", ("小猫", "小狗", "风景", "截图", "宠物")),
    QuerySpec("car", 2, "汽车图片", GENERIC_NEGATIVES),
    QuerySpec(
        "cat",
        15,
        "小猫图片",
        ("其它动物照片", "狗的照片", "野生动物照片", "老虎照片", "风景照片", "人物照片", "手机截图"),
    ),
    QuerySpec(
        "dog",
        16,
        "小狗图片",
        ("猫的照片", "野生动物照片", "老虎照片", "风景照片", "人物照片", "手机截图"),
    ),
    QuerySpec("handbag", 26, "手提包图片", GENERIC_NEGATIVES),
    QuerySpec("cup", 41, "杯子图片", GENERIC_NEGATIVES),
    QuerySpec("bowl", 45, "碗的图片", GENERIC_NEGATIVES),
    QuerySpec("chair", 56, "椅子图片", GENERIC_NEGATIVES),
    QuerySpec("potted_plant", 58, "盆栽图片", GENERIC_NEGATIVES),
    QuerySpec("clock", 74, "时钟图片", GENERIC_NEGATIVES),
)


def load_coco128() -> list[tuple[str, set[int], Image.Image]]:
    ensure_coco_archive()
    samples: list[tuple[str, set[int], Image.Image]] = []
    empty_label_count = 0
    with zipfile.ZipFile(COCO_ARCHIVE_PATH) as archive:
        image_names = sorted(
            name
            for name in archive.namelist()
            if name.startswith(COCO_IMAGE_PREFIX) and name.endswith(".jpg")
        )
        if len(image_names) != EXPECTED_IMAGE_COUNT:
            raise SystemExit(
                f"COCO128 image count mismatch: expected {EXPECTED_IMAGE_COUNT}, "
                f"found {len(image_names)}"
            )

        archive_names = set(archive.namelist())
        for image_name in image_names:
            image_id = Path(image_name).stem
            label_name = f"{COCO_LABEL_PREFIX}{image_id}.txt"
            if label_name in archive_names:
                labels = {
                    int(line.split(maxsplit=1)[0])
                    for line in archive.read(label_name).decode("utf-8").splitlines()
                    if line.strip()
                }
            else:
                labels = set()
                empty_label_count += 1
            with Image.open(io.BytesIO(archive.read(image_name))) as image:
                samples.append((image_id, labels, image.convert("RGB").copy()))
    if empty_label_count != EXPECTED_EMPTY_LABEL_COUNT:
        raise SystemExit(
            "COCO128 empty-label count mismatch: "
            f"expected {EXPECTED_EMPTY_LABEL_COUNT}, found {empty_label_count}"
        )
    return samples


def coarse_image(image: Image.Image) -> Image.Image:
    coarse = image.copy()
    coarse.thumbnail((COARSE_SIDE, COARSE_SIDE), Image.Resampling.BICUBIC)
    return coarse


def encode_images(model, preprocess, images: list[Image.Image]) -> torch.Tensor:
    encoded: list[torch.Tensor] = []
    for start in range(0, len(images), BATCH_SIZE):
        batch = torch.stack([preprocess(image) for image in images[start : start + BATCH_SIZE]])
        with torch.inference_mode():
            encoded.append(normalized(model.encode_image(batch)))
    return torch.cat(encoded)


def text_embeddings(model, clip) -> dict[str, torch.Tensor]:
    prompts: list[str] = []
    for query in QUERIES:
        for prompt in (query.positive, *query.negatives):
            if prompt not in prompts:
                prompts.append(prompt)
    with torch.inference_mode():
        embeddings = normalized(model.encode_text(clip.tokenize(prompts)))
    return {prompt: embeddings[index] for index, prompt in enumerate(prompts)}


def rank_scores(
    image_features: torch.Tensor,
    query: QuerySpec,
    text_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    positive = image_features @ text_features[query.positive]
    negatives = torch.stack([text_features[prompt] for prompt in query.negatives])
    strongest_negative = (image_features @ negatives.T).max(dim=1).values
    margin = positive - strongest_negative
    return margin * 1_000 + positive * 10


def ranking_metrics(
    ranking: torch.Tensor,
    labels: list[set[int]],
    class_id: int,
) -> tuple[int, float]:
    order = torch.argsort(ranking, descending=True)
    top1_hit = int(class_id in labels[int(order[0])])
    precision_at_5 = sum(class_id in labels[int(index)] for index in order[:5]) / 5
    return top1_hit, precision_at_5


def main() -> None:
    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint_path, _ = validate_checkpoint(model_manifest)
    clip = load_cn_clip(ROOT / model_manifest["source"]["path"])

    from cn_clip.clip.utils import _MODEL_INFO, create_model, image_transform

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(_MODEL_INFO["RN50"]["struct"], checkpoint).float().eval()
    preprocess = image_transform(224)
    samples = load_coco128()
    image_ids = [image_id for image_id, _, _ in samples]
    labels = [sample_labels for _, sample_labels, _ in samples]

    missing_classes = [
        query.name
        for query in QUERIES
        if not any(query.class_id in item for item in labels)
    ]
    if missing_classes:
        raise SystemExit(f"COCO128 target classes are missing: {missing_classes}")

    high_quality_features = encode_images(model, preprocess, [image for _, _, image in samples])
    coarse_features = encode_images(
        model,
        preprocess,
        [coarse_image(image) for _, _, image in samples],
    )
    text_features = text_embeddings(model, clip)

    high_top1_hits = 0
    high_precision_at_5 = 0.0
    coarse_top1_hits = 0
    coarse_precision_at_5 = 0.0
    shortlist_recalls: list[float] = []

    for query in QUERIES:
        high_scores = rank_scores(high_quality_features, query, text_features)
        coarse_scores = rank_scores(coarse_features, query, text_features)
        high_top1, high_p5 = ranking_metrics(high_scores, labels, query.class_id)
        coarse_top1, coarse_p5 = ranking_metrics(coarse_scores, labels, query.class_id)
        high_top1_hits += high_top1
        high_precision_at_5 += high_p5
        coarse_top1_hits += coarse_top1
        coarse_precision_at_5 += coarse_p5

        oracle_indices = torch.argsort(high_scores, descending=True)[:ORACLE_TOP_K]
        shortlist_indices = set(
            int(index) for index in torch.argsort(coarse_scores, descending=True)[:SHORTLIST_K]
        )
        retained = sum(int(index) in shortlist_indices for index in oracle_indices)
        recall = retained / ORACLE_TOP_K
        shortlist_recalls.append(recall)
        missing_ids = [
            image_ids[int(index)]
            for index in oracle_indices
            if int(index) not in shortlist_indices
        ]
        print(
            f"query={query.name} highTop1={high_top1} highP@5={high_p5:.3f} "
            f"coarseTop1={coarse_top1} coarseP@5={coarse_p5:.3f} "
            f"oracleTop{ORACLE_TOP_K}Recall@{SHORTLIST_K}={recall:.3f} "
            f"missing={','.join(missing_ids) if missing_ids else 'none'}"
        )

    query_count = len(QUERIES)
    high_macro_p5 = high_precision_at_5 / query_count
    coarse_macro_p5 = coarse_precision_at_5 / query_count
    macro_shortlist_recall = sum(shortlist_recalls) / query_count
    minimum_shortlist_recall = min(shortlist_recalls)
    print(
        f"highQualityFull Top1={high_top1_hits}/{query_count} macroP@5={high_macro_p5:.3f}"
    )
    print(
        f"coarse{COARSE_SIDE} Top1={coarse_top1_hits}/{query_count} "
        f"macroP@5={coarse_macro_p5:.3f}"
    )
    print(
        f"coarse{COARSE_SIDE} oracleTop{ORACLE_TOP_K}Recall@{SHORTLIST_K} "
        f"macro={macro_shortlist_recall:.3f} minQuery={minimum_shortlist_recall:.3f}"
    )

    failures = []
    if high_top1_hits < MINIMUM_TOP1_HITS:
        failures.append(
            f"high-quality Top1 regressed: expected at least {MINIMUM_TOP1_HITS}/{query_count}, "
            f"found {high_top1_hits}/{query_count}"
        )
    if high_macro_p5 < MINIMUM_MACRO_PRECISION_AT_5:
        failures.append(
            "high-quality macro P@5 regressed: "
            f"expected at least {MINIMUM_MACRO_PRECISION_AT_5:.3f}, found {high_macro_p5:.3f}"
        )
    if minimum_shortlist_recall < 1.0:
        failures.append(
            f"coarse Top-{SHORTLIST_K} did not retain every high-quality Top-{ORACLE_TOP_K} result"
        )
    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    print("PASS: multiclass ranking quality and coarse shortlist recall are within gates")


if __name__ == "__main__":
    main()
