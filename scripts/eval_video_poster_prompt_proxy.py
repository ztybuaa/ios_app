import argparse
import json
from pathlib import Path

from env_guard import ensure_project_venv


ensure_project_venv()

import torch

from diagnose_rn50_precision import (
    COCO_ARCHIVE_BYTES,
    COCO_ARCHIVE_PATH,
    COCO_ARCHIVE_SHA256,
    COCO_ARCHIVE_URL,
)
from eval_chinese_clip_rn50 import (
    MANIFEST_PATH,
    MODEL_MANIFEST_PATH,
    ROOT,
    load_cn_clip,
    normalized,
    validate_checkpoint,
)
from validate_chinese_clip_multiclass_quality import (
    QUERIES,
    encode_images,
    load_coco128,
)

PROMPT_VARIANTS = ("subject_image", "subject_video", "subject_only")
SUBJECT_SUFFIXES = ("的图片", "图片", "的照片", "照片")
SCORE_SCALE = 1_000
POSITIVE_TIE_BREAK_SCALE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Chinese-CLIP prompt surfaces on the existing COCO128 "
            "multiclass fixtures as a video-poster retrieval proxy."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination for the deterministic JSON report.",
    )
    return parser.parse_args()


def subject_from_image_prompt(prompt: str) -> str:
    for suffix in SUBJECT_SUFFIXES:
        if prompt.endswith(suffix):
            subject = prompt[: -len(suffix)]
            if subject:
                return subject
    raise SystemExit(
        f"Multiclass query prompt does not have a recognized image suffix: {prompt!r}"
    )


def prompt_for_variant(original_prompt: str, variant: str) -> str:
    subject = subject_from_image_prompt(original_prompt)
    if variant == "subject_image":
        return original_prompt
    if variant == "subject_video":
        return f"{subject}视频"
    if variant == "subject_only":
        return subject
    raise AssertionError(f"Unknown prompt variant: {variant}")


def encode_texts(model, clip) -> dict[str, torch.Tensor]:
    prompts: list[str] = []
    for variant in PROMPT_VARIANTS:
        for query in QUERIES:
            prompt = prompt_for_variant(query.positive, variant)
            if prompt not in prompts:
                prompts.append(prompt)
            for negative in query.negatives:
                if negative not in prompts:
                    prompts.append(negative)

    with torch.inference_mode():
        features = normalized(model.encode_text(clip.tokenize(prompts)))
    return {prompt: features[index] for index, prompt in enumerate(prompts)}


def round_metric(value: float) -> float:
    return round(value, 6)


def evaluate_variant(
    variant: str,
    image_features: torch.Tensor,
    image_ids: list[str],
    labels: list[set[int]],
    text_features: dict[str, torch.Tensor],
    minimum_similarity: float,
    minimum_margin: float,
) -> dict:
    query_results = []
    top1_hits = 0
    precision_at_5_sum = 0.0
    gate_true_positives = 0
    gate_false_positives = 0
    gate_query_passes = 0
    gate_query_precisions: list[float] = []

    for query in QUERIES:
        prompt = prompt_for_variant(query.positive, variant)
        positive_scores = image_features @ text_features[prompt]
        negative_features = torch.stack(
            [text_features[negative] for negative in query.negatives]
        )
        negative_scores = (image_features @ negative_features.T).max(dim=1).values
        margins = positive_scores - negative_scores
        rank_scores = margins * SCORE_SCALE + positive_scores * POSITIVE_TIE_BREAK_SCALE
        ranking = sorted(
            range(len(image_ids)),
            key=lambda index: (
                -float(rank_scores[index].item()),
                -float(positive_scores[index].item()),
                image_ids[index],
            ),
        )

        top1_hit = query.class_id in labels[ranking[0]]
        precision_at_5 = sum(
            query.class_id in labels[index] for index in ranking[:5]
        ) / 5
        top1_hits += int(top1_hit)
        precision_at_5_sum += precision_at_5

        passing = [
            index
            for index in range(len(image_ids))
            if float(positive_scores[index].item()) >= minimum_similarity
            and float(margins[index].item()) >= minimum_margin
        ]
        true_positives = sum(query.class_id in labels[index] for index in passing)
        false_positives = len(passing) - true_positives
        gate_true_positives += true_positives
        gate_false_positives += false_positives
        gate_query_pass = true_positives > 0 and false_positives == 0
        gate_query_passes += int(gate_query_pass)
        gate_precision = (
            true_positives / len(passing) if passing else None
        )
        if gate_precision is not None:
            gate_query_precisions.append(gate_precision)

        query_results.append(
            {
                "id": query.name,
                "classID": query.class_id,
                "subject": subject_from_image_prompt(query.positive),
                "prompt": prompt,
                "top1Hit": top1_hit,
                "precisionAt5": round_metric(precision_at_5),
                "thresholdGate": {
                    "passingCount": len(passing),
                    "truePositiveCount": true_positives,
                    "falsePositiveCount": false_positives,
                    "precision": (
                        round_metric(gate_precision)
                        if gate_precision is not None
                        else None
                    ),
                    "queryPass": gate_query_pass,
                },
                "top5": [
                    {
                        "imageID": image_ids[index],
                        "relevant": query.class_id in labels[index],
                    }
                    for index in ranking[:5]
                ],
            }
        )

    query_count = len(QUERIES)
    gate_passing_count = gate_true_positives + gate_false_positives
    gate_micro_precision = (
        gate_true_positives / gate_passing_count if gate_passing_count else None
    )
    gate_macro_precision = (
        sum(gate_query_precisions) / len(gate_query_precisions)
        if gate_query_precisions
        else None
    )
    return {
        "id": variant,
        "top1": {
            "hits": top1_hits,
            "queryCount": query_count,
            "rate": round_metric(top1_hits / query_count),
        },
        "macroPrecisionAt5": round_metric(precision_at_5_sum / query_count),
        "thresholdGatePrecision": {
            "minimumSimilarity": minimum_similarity,
            "minimumMargin": minimum_margin,
            "passingCount": gate_passing_count,
            "truePositiveCount": gate_true_positives,
            "falsePositiveCount": gate_false_positives,
            "microPrecision": (
                round_metric(gate_micro_precision)
                if gate_micro_precision is not None
                else None
            ),
            "macroPrecisionOverQueriesWithPassing": (
                round_metric(gate_macro_precision)
                if gate_macro_precision is not None
                else None
            ),
            "queriesWithPassing": len(gate_query_precisions),
            "queriesWithoutPassing": query_count - len(gate_query_precisions),
            "queryPasses": gate_query_passes,
            "queryPassRate": round_metric(gate_query_passes / query_count),
        },
        "queries": query_results,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    model_manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    semantic_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint_path, checkpoint_sha256 = validate_checkpoint(model_manifest)
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

    image_features = encode_images(
        model,
        preprocess,
        [image for _, _, image in samples],
    )
    text_features = encode_texts(model, clip)
    minimum_similarity = float(
        semantic_manifest["calibration"]["minimumSimilarity"]
    )
    minimum_margin = float(
        semantic_manifest["calibration"]["standardMinimumMargin"]
    )
    variants = [
        evaluate_variant(
            variant,
            image_features,
            image_ids,
            labels,
            text_features,
            minimum_similarity,
            minimum_margin,
        )
        for variant in PROMPT_VARIANTS
    ]
    by_id = {item["id"]: item for item in variants}
    image_variant = by_id["subject_image"]
    video_variant = by_id["subject_video"]
    image_precision = image_variant["thresholdGatePrecision"]["microPrecision"]
    video_precision = video_variant["thresholdGatePrecision"]["microPrecision"]
    if image_variant["top1"]["rate"] < 0.60 or image_precision is None or image_precision < 0.70:
        raise SystemExit("subject+image video-poster proxy regressed below its calibrated floor")
    if image_variant["top1"]["rate"] < video_variant["top1"]["rate"]:
        raise SystemExit("subject+image Top-1 unexpectedly fell below subject+video")
    if video_precision is not None and image_precision < video_precision:
        raise SystemExit("subject+image gate precision unexpectedly fell below subject+video")

    report = {
        "schemaVersion": 1,
        "experiment": "video_poster_prompt_proxy",
        "hypothesis": (
            "A video-specific Chinese prompt preserves subject retrieval quality when "
            "video poster frames are embedded by the existing Chinese-CLIP RN50 model."
        ),
        "controls": {
            "changedVariable": "positive prompt surface only",
            "fixedNegativePrompts": True,
            "fixedRanking": "margin*1000 + positiveSimilarity*10",
            "fixedThresholdGate": True,
            "imageView": "full",
        },
        "model": {
            "id": model_manifest["model"]["id"],
            "architecture": model_manifest["model"]["architecture"],
            "checkpointSHA256": checkpoint_sha256,
        },
        "dataset": {
            "name": "COCO128",
            "archive": COCO_ARCHIVE_PATH.relative_to(ROOT).as_posix(),
            "source": COCO_ARCHIVE_URL,
            "bytes": COCO_ARCHIVE_BYTES,
            "sha256": COCO_ARCHIVE_SHA256,
            "imageCount": len(samples),
            "queryCount": len(QUERIES),
            "querySource": "scripts/validate_chinese_clip_multiclass_quality.py",
        },
        "variants": variants,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for result in variants:
        gate = result["thresholdGatePrecision"]
        micro_precision = gate["microPrecision"]
        rendered_precision = (
            f"{micro_precision:.3f}" if micro_precision is not None else "undefined"
        )
        print(
            f"{result['id']}: Top1={result['top1']['hits']}/"
            f"{result['top1']['queryCount']} "
            f"macroP@5={result['macroPrecisionAt5']:.3f} "
            f"gatePrecision={rendered_precision} "
            f"gateTP={gate['truePositiveCount']} gateFP={gate['falsePositiveCount']} "
            f"gateQueries={gate['queryPasses']}/{result['top1']['queryCount']}"
        )
    print(f"JSON report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
