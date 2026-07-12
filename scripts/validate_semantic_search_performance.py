import argparse
import re
from pathlib import Path

from env_guard import ensure_project_venv


ensure_project_venv()

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "ios_app" / "IntentResourceDemo" / "ResourceModules" / "SemanticImageSearchService.swift"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VALIDATION FAILED: {message}")


def parse_policy(source: str) -> tuple[int, int, int]:
    scan_start = source.index("private func semanticScanLimit")
    scan_end = source.index("private func", scan_start + 1)
    scan_limits = re.findall(r"return\s+(\d[\d_]*)", source[scan_start:scan_end])
    require(scan_limits, "default semantic scan limit is missing")
    default_scan_limit = int(scan_limits[-1].replace("_", ""))

    rerank_match = re.search(r"max\((\d+),\s*resultLimit\s*\*\s*(\d+)\)", source)
    require(rerank_match is not None, "bounded semantic rerank policy is missing")
    rerank_floor = int(rerank_match.group(1))
    rerank_multiplier = int(rerank_match.group(2))

    require("profile: .full" in source, "full-image coarse pass is missing")
    require("profile: .regions" in source, "regional shortlist rerank is missing")
    require("ImageEmbeddingStore(namespace:" in source, "persistent embedding store is missing")
    require("applicationSupportDirectory" in source, "persistent embeddings are not stored in Application Support")
    require(
        "needsRegionScan: !wantsCat && !wantsDog" in source,
        "animal queries must remain full-image only",
    )
    require("\"other animal photos\"" not in source, "English prompt leaked into native Chinese retrieval")
    require("\"\u5176\u5b83\u52a8\u7269\u7167\u7247\"" in source, "cat hard-negative precision prompt is missing")
    return default_scan_limit, rerank_floor, rerank_multiplier


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the bounded two-stage iOS semantic indexing policy.")
    parser.add_argument("--assets", type=int, default=2_000)
    parser.add_argument("--result-limit", type=int, default=12)
    parser.add_argument("--image-view-ms", type=float)
    parser.add_argument("--budget-seconds", type=float)
    parser.add_argument("--query-profile", choices=("cat", "regional"), default="regional")
    args = parser.parse_args()

    require(args.assets >= 0, "asset count must be non-negative")
    require(args.result_limit > 0, "result limit must be positive")
    require(
        (args.image_view_ms is None) == (args.budget_seconds is None),
        "image-view latency and latency budget must be provided together",
    )
    if args.image_view_ms is not None:
        require(args.image_view_ms >= 0, "image-view latency must be non-negative")
        require(args.budget_seconds > 0, "latency budget must be positive")
    source = SERVICE.read_text(encoding="utf-8")
    scan_limit, rerank_floor, rerank_multiplier = parse_policy(source)

    scanned_assets = min(args.assets, scan_limit)
    shortlist = 0
    region_predictions = 0
    if args.query_profile == "regional":
        shortlist = min(scanned_assets, max(rerank_floor, args.result_limit * rerank_multiplier))
    region_predictions = shortlist * 5
    full_predictions = scanned_assets
    total_predictions = full_predictions + region_predictions
    exhaustive_predictions = scanned_assets * 6
    reduction = 0.0 if exhaustive_predictions == 0 else 1 - total_predictions / exhaustive_predictions

    print(f"query_profile={args.query_profile}")
    print(f"assets_scanned={scanned_assets}")
    print(f"shortlist_assets={shortlist}")
    print(f"full_predictions={full_predictions}")
    print(f"region_predictions={region_predictions}")
    print(f"total_predictions={total_predictions}")
    print(f"exhaustive_six_view_predictions={exhaustive_predictions}")
    print(f"prediction_reduction_percent={reduction * 100:.1f}")

    require(
        total_predictions <= exhaustive_predictions / 4 if scanned_assets else True,
        "the two-stage policy does not materially reduce six-view full-album inference",
    )
    if args.image_view_ms is not None:
        estimated_seconds = total_predictions * args.image_view_ms / 1_000
        print(f"scenario_model_seconds={estimated_seconds:.3f}")
        print(f"scenario_budget_seconds={args.budget_seconds:.3f}")
        require(estimated_seconds <= args.budget_seconds, "model-only latency scenario exceeds the budget")
    print("Semantic indexing performance validation passed")


if __name__ == "__main__":
    main()
