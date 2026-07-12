import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from env_guard import ensure_project_venv


ensure_project_venv()

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "processed" / "eval" / "semantic_image_retrieval"
IMAGE_ROOT = DATASET_ROOT / "images"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
RESULT_PATH = DATASET_ROOT / "results" / "chinese_clip_rn50_eval_report.json"
REPORT_PATH = ROOT / "reports" / "chinese_clip_rn50_fp16_eval.md"
MODEL_ROOT = ROOT / "external_models" / "pretrained" / "chinese_clip_rn50"
CHECKPOINT_PATH = MODEL_ROOT / "clip_cn_rn50.pt"
SOURCE_ROOT = MODEL_ROOT / "source" / "Chinese-CLIP"

EXPECTED_CHECKPOINT_BYTES = 308_316_425
EXPECTED_CHECKPOINT_SHA256 = "b196ee3ee528b70be1158ab1aafb1d2f1c801ad2d9ffb3bae31b0d305f82fc88"
def load_cn_clip():
    if SOURCE_ROOT.exists():
        sys.path.insert(0, str(SOURCE_ROOT))
        source = str(SOURCE_ROOT.relative_to(ROOT))
    else:
        source = "installed cn_clip package"

    try:
        import cn_clip.clip as clip
    except ImportError as error:
        raise SystemExit(
            "Chinese-CLIP evaluation dependency is missing. Install the pinned evaluation "
            "dependencies in the project .venv before running this script."
        ) from error

    print(f"Chinese-CLIP implementation: {source}")
    return clip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint() -> str:
    if not CHECKPOINT_PATH.is_file():
        raise SystemExit(f"Chinese-CLIP checkpoint is missing: {CHECKPOINT_PATH}")
    actual_bytes = CHECKPOINT_PATH.stat().st_size
    if actual_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise SystemExit(
            f"Chinese-CLIP checkpoint size mismatch: expected {EXPECTED_CHECKPOINT_BYTES}, "
            f"found {actual_bytes}"
        )
    actual_sha256 = sha256(CHECKPOINT_PATH)
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise SystemExit(
            f"Chinese-CLIP checkpoint SHA-256 mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, "
            f"found {actual_sha256}"
        )
    return actual_sha256


def image_views(image: Image.Image, exact_ios_views: bool) -> list[tuple[str, Image.Image]]:
    image = image.convert("RGB")
    if not exact_ios_views:
        return [("full", image)]

    width, height = image.size
    side = min(width, height)
    center_x = (width - side) // 2
    center_y = (height - side) // 2
    boxes = {
        "center": (center_x, center_y, center_x + side, center_y + side),
        "top-left": (0, 0, side, side),
        "top-right": (width - side, 0, width, side),
        "bottom-left": (0, height - side, side, height),
        "bottom-right": (width - side, height - side, width, height),
    }
    return [("full", image)] + [(name, image.crop(box)) for name, box in boxes.items()]


def normalized(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True)


def evaluate_mode(model, preprocess, clip, manifest: dict, exact_ios_views: bool) -> dict:
    mode_name = "ios_exact_views" if exact_ios_views else "full_only"
    tensors = []
    image_view_metadata: list[tuple[str, str]] = []

    for image_entry in manifest["images"]:
        image_path = IMAGE_ROOT / image_entry["file"]
        if not image_path.is_file():
            raise SystemExit(f"Evaluation image is missing: {image_path}")
        with Image.open(image_path) as image:
            for view_name, view in image_views(image, exact_ios_views):
                tensors.append(preprocess(view))
                image_view_metadata.append((image_entry["id"], view_name))

    started_at = time.perf_counter()
    with torch.inference_mode():
        image_features = normalized(model.encode_image(torch.stack(tensors)))
    image_inference_ms = (time.perf_counter() - started_at) * 1_000

    query_results = []
    top_k_hits = 0
    precision_gate_passes = 0
    false_positive_count = 0

    for query in manifest["queries"]:
        query_id = query["id"]
        positive_prompt = query["chinese"]
        negative_prompts = query.get("hardNegatives", [])
        if not negative_prompts:
            raise SystemExit(f"Hard-negative prompts are missing for query {query_id}")
        text_inputs = clip.tokenize([positive_prompt] + negative_prompts)
        with torch.inference_mode():
            text_features = normalized(model.encode_text(text_inputs))

        positive_scores = image_features @ text_features[0]
        negative_scores = (image_features @ text_features[1:].T).max(dim=1).values
        margins = positive_scores - negative_scores

        best_by_image: dict[str, dict] = {}
        for index, (image_id, view_name) in enumerate(image_view_metadata):
            candidate = {
                "imageID": image_id,
                "view": view_name,
                "positive": float(positive_scores[index].item()),
                "negative": float(negative_scores[index].item()),
                "margin": float(margins[index].item()),
            }
            previous = best_by_image.get(image_id)
            if previous is None or (candidate["margin"], candidate["positive"]) > (
                previous["margin"],
                previous["positive"],
            ):
                best_by_image[image_id] = candidate

        ranking = sorted(
            best_by_image.values(),
            key=lambda candidate: (candidate["margin"], candidate["positive"], candidate["imageID"]),
            reverse=True,
        )
        expected = set(query["expected"])
        top_k = int(query["topK"])
        top_k_hit = any(candidate["imageID"] in expected for candidate in ranking[:top_k])
        if top_k_hit:
            top_k_hits += 1

        minimum_similarity = float(manifest["calibration"]["minimumSimilarity"])
        margin_threshold = float(query["minimumMargin"])
        passing = [
            candidate
            for candidate in ranking
            if candidate["positive"] >= minimum_similarity
            and candidate["margin"] >= margin_threshold
        ]
        expected_passing = [candidate for candidate in passing if candidate["imageID"] in expected]
        false_positives = [candidate for candidate in passing if candidate["imageID"] not in expected]
        precision_gate_pass = bool(expected_passing) and not false_positives
        if precision_gate_pass:
            precision_gate_passes += 1
        false_positive_count += len(false_positives)

        query_results.append(
            {
                "id": query_id,
                "chinese": positive_prompt,
                "expected": sorted(expected),
                "topK": top_k,
                "topKHit": top_k_hit,
                "minimumSimilarity": minimum_similarity,
                "minimumMargin": margin_threshold,
                "hardNegatives": negative_prompts,
                "precisionGatePass": precision_gate_pass,
                "passing": passing,
                "falsePositives": false_positives,
                "ranking": ranking,
            }
        )

    query_count = len(manifest["queries"])
    return {
        "mode": mode_name,
        "imageViewCount": len(image_view_metadata),
        "imageInferenceMs": image_inference_ms,
        "queryCount": query_count,
        "topKHits": top_k_hits,
        "precisionGatePasses": precision_gate_passes,
        "falsePositiveCount": false_positive_count,
        "passed": top_k_hits == query_count
        and precision_gate_passes == query_count
        and false_positive_count == 0,
        "queries": query_results,
    }


def write_markdown(report: dict) -> None:
    lines = [
        "# Chinese-CLIP RN50 FP16 迁移前评测",
        "",
        "## 结论",
        "",
        "本报告使用官方 RN50 PyTorch checkpoint 评估原生中文检索，并用相同提示词和裁剪规则校准待转换的 FP16 Core ML 模型。Core ML 数值一致性由 macOS 转换流程单独验证。",
        "",
        f"- checkpoint SHA-256: `{report['checkpoint']['sha256']}`",
        f"- 中文查询数: `{report['dataset']['queryCount']}`",
        f"- 图片数: `{report['dataset']['imageCount']}`",
        f"- minimum similarity: `{report['calibration']['minimumSimilarity']:.3f}`",
        f"- 普通查询 minimum margin: `{report['calibration']['standardMinimumMargin']:.3f}`",
        f"- 普通截图 minimum margin: `{report['calibration']['screenshotMinimumMargin']:.3f}`",
        "",
        "## 结果",
        "",
        "| 模式 | Top-K | 高精度门限 | 已知误检 | 结论 |",
        "|---|---:|---:|---:|---|",
    ]
    for mode in report["modes"]:
        lines.append(
            f"| {mode['mode']} | {mode['topKHits']}/{mode['queryCount']} | "
            f"{mode['precisionGatePasses']}/{mode['queryCount']} | "
            f"{mode['falsePositiveCount']} | {'通过' if mode['passed'] else '失败'} |"
        )

    exact_mode = next(mode for mode in report["modes"] if mode["mode"] == "ios_exact_views")
    lines.extend(
        [
            "",
            "## iOS 六视图明细",
            "",
            "| 查询 | 通过候选 | Top-3 |",
            "|---|---|---|",
        ]
    )
    for query in exact_mode["queries"]:
        passing = ", ".join(
            f"{candidate['imageID']} ({candidate['view']}, m={candidate['margin']:.4f})"
            for candidate in query["passing"]
        ) or "无"
        top_three = ", ".join(
            f"{candidate['imageID']} ({candidate['view']}, m={candidate['margin']:.4f})"
            for candidate in query["ranking"][:3]
        )
        lines.append(f"| {query['chinese']} | {passing} | {top_three} |")

    lines.extend(
        [
            "",
            "## 限制",
            "",
            "当前集合只有 10 张图片和 6 条查询，足以拦截已知猫狗、风景、人物、截图和游戏截图回归，但不能替代真实相册评测。最终门限还需结合 FP16 Core ML 数值一致性和真机相册结果复核。",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    checkpoint_sha256 = validate_checkpoint()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    clip = load_cn_clip()

    from cn_clip.clip.utils import _MODEL_INFO, create_model, image_transform

    checkpoint_payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = create_model(_MODEL_INFO["RN50"]["struct"], checkpoint_payload).float().eval()
    preprocess = image_transform(224)

    modes = [
        evaluate_mode(model, preprocess, clip, manifest, exact_ios_views=False),
        evaluate_mode(model, preprocess, clip, manifest, exact_ios_views=True),
    ]
    report = {
        "version": 1,
        "model": "Chinese-CLIP RN50",
        "targetPrecision": "Core ML FP16",
        "checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "bytes": CHECKPOINT_PATH.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "dataset": {
            "manifest": str(MANIFEST_PATH.relative_to(ROOT)),
            "imageCount": len(manifest["images"]),
            "queryCount": len(manifest["queries"]),
        },
        "calibration": manifest["calibration"],
        "modes": modes,
        "passed": all(mode["passed"] for mode in modes),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(report)

    for mode in modes:
        status = "PASS" if mode["passed"] else "FAIL"
        print(
            f"[{status}] {mode['mode']}: topK={mode['topKHits']}/{mode['queryCount']} "
            f"precisionGate={mode['precisionGatePasses']}/{mode['queryCount']} "
            f"falsePositives={mode['falsePositiveCount']}"
        )
        for query in mode["queries"]:
            passing = ", ".join(
                f"{candidate['imageID']}/{candidate['view']} m={candidate['margin']:.4f}"
                for candidate in query["passing"]
            ) or "none"
            print(f"  {query['chinese']}: {passing}")

    print(f"JSON report: {RESULT_PATH.relative_to(ROOT)}")
    print(f"Markdown report: {REPORT_PATH.relative_to(ROOT)}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
