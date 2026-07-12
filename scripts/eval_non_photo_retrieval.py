#!/usr/bin/env python3
"""Compare the current flat substring scorer with a field-aware lexical scorer.

The fixture is synthetic and deterministic. The proposed scorer intentionally uses
only Python's standard library so this pre-device regression can run offline.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "processed" / "eval" / "non_photo_resource_retrieval" / "manifest.json"

SUPPORTED_KINDS = ("file", "folder", "contact")
RESOURCE_TERMS = {
    "file": ("Word文件", "Excel表格", "PDF文件", "PPT文件", "文件", "文档", "表格"),
    "folder": ("文件夹", "资料夹", "目录"),
    "contact": (
        "联系人信息",
        "联系方式",
        "电话号码",
        "邮箱地址",
        "手机号码",
        "联系人",
        "通讯录",
        "手机号",
        "电话",
        "邮箱",
        "名片",
    ),
}
TIME_TERMS = ("刚更新", "刚保存", "最近的", "最新的", "最近", "最新", "今天", "昨天", "本周", "上周")
PARTICLES = ("请帮我", "帮我", "查找", "搜索", "打开", "找到", "选择", "发给", "发送", "找", "把", "给我")
FORMAT_ALIASES = {
    "pdf": ("pdf",),
    "word": ("word", "docx", "doc"),
    "excel": ("excel", "xlsx", "xls"),
    "ppt": ("powerpoint", "pptx", "ppt"),
    "text": ("text", "txt"),
    "zip": ("zip",),
}
FIELD_WEIGHTS = {
    "file": {
        "title": 8.0,
        "tags": 5.0,
        "content_terms": 4.0,
        "summary": 3.5,
        "path": 2.0,
    },
    "folder": {
        "title": 8.0,
        "tags": 5.0,
        "child_terms": 4.5,
        "summary": 3.5,
        "path": 2.0,
    },
    "contact": {
        "name": 8.0,
        "nickname": 7.0,
        "phonetic": 7.0,
        "job_title": 5.0,
        "organization": 4.5,
    },
}
MIN_LEXICAL_SCORE = 3.0
MIN_FIELD_SIMILARITY = 0.45


@dataclass(frozen=True)
class QueryPlan:
    subject: str
    formats: frozenset[str]
    recent: bool
    updated_after: date | None
    updated_before: date | None


def normalize_display_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def compact_text(value: str) -> str:
    return "".join(character for character in normalize_display_text(value) if character.isalnum())


def digits_only(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def values(candidate: dict[str, Any], field: str) -> list[str]:
    raw = candidate.get(field, [])
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    return [str(raw)] if str(raw) else []


def canonical_format(value: str) -> str:
    normalized = compact_text(value)
    for canonical, aliases in FORMAT_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def detected_formats(text: str) -> set[str]:
    normalized = normalize_display_text(text)
    found: set[str] = set()
    for canonical, aliases in FORMAT_ALIASES.items():
        if any(re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized) for alias in aliases):
            found.add(canonical)
    return found


def parse_optional_date(raw: Any) -> date | None:
    return date.fromisoformat(str(raw)) if raw else None


def build_query_plan(query: dict[str, Any]) -> QueryPlan:
    keyword = str(query.get("keyword") or "").strip()
    phrase = str(query.get("phrase") or "").strip()
    subject = normalize_display_text(keyword or phrase)
    qualifiers = query.get("qualifiers", {})

    formats = {canonical_format(str(item)) for item in qualifiers.get("format", [])}
    formats.update(detected_formats(subject))
    selection_hints = {str(item) for item in qualifiers.get("selection_hint", [])}
    recent = "recent" in selection_hints or any(term in subject for term in TIME_TERMS)

    removal_terms: list[str] = []
    removal_terms.extend(RESOURCE_TERMS[query["kind"]])
    removal_terms.extend(TIME_TERMS)
    removal_terms.extend(PARTICLES)
    for aliases in FORMAT_ALIASES.values():
        removal_terms.extend(aliases)
    for term in sorted(removal_terms, key=len, reverse=True):
        subject = subject.replace(normalize_display_text(term), "")
    subject = re.sub(r"\s+", " ", subject.replace("的", " "))
    subject = subject.strip(" ，,。.;；:：_-／/")

    return QueryPlan(
        subject=subject,
        formats=frozenset(formats),
        recent=recent,
        updated_after=parse_optional_date(qualifiers.get("updated_after")),
        updated_before=parse_optional_date(qualifiers.get("updated_before")),
    )


def ngrams(text: str, size: int) -> set[str]:
    if len(text) <= size:
        return {text} if text else set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def lexical_similarity(query_text: str, candidate_text: str) -> float:
    query = compact_text(query_text)
    candidate = compact_text(candidate_text)
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0
    if query in candidate:
        return min(0.99, 0.85 + 0.15 * len(query) / len(candidate))
    if candidate in query and len(candidate) >= 2:
        return 0.9 * len(candidate) / len(query)

    uses_cjk = any("\u4e00" <= character <= "\u9fff" for character in query)
    size = 2 if uses_cjk or len(query) < 5 else 3
    query_grams = ngrams(query, size)
    if not query_grams:
        return 0.0
    coverage = len(query_grams & ngrams(candidate, size)) / len(query_grams)
    return 0.8 * coverage if coverage >= 0.30 else 0.0


def specialized_contact_score(subject: str, candidate: dict[str, Any]) -> float:
    query_digits = digits_only(subject)
    if len(query_digits) >= 7:
        for phone in values(candidate, "phones"):
            candidate_digits = digits_only(phone)
            if query_digits == candidate_digits:
                return 10.0
            if len(query_digits) >= 7 and candidate_digits.endswith(query_digits):
                return 8.0
        return 0.0

    email_match = re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalize_display_text(subject))
    if email_match:
        query_email = email_match.group(0)
        return 10.0 if query_email in {normalize_display_text(item) for item in values(candidate, "emails")} else 0.0
    return -1.0


def field_aware_lexical_score(plan: QueryPlan, candidate: dict[str, Any]) -> float:
    if candidate["kind"] == "contact":
        specialized = specialized_contact_score(plan.subject, candidate)
        if specialized >= 0:
            return specialized

    query_fragments = list(dict.fromkeys([plan.subject, *plan.subject.split()]))
    matches: list[float] = []
    for field, weight in FIELD_WEIGHTS[candidate["kind"]].items():
        field_match = 0.0
        for field_value in values(candidate, field):
            similarity = max(lexical_similarity(fragment, field_value) for fragment in query_fragments)
            if similarity >= MIN_FIELD_SIMILARITY:
                field_match = max(field_match, weight * similarity)
        if field_match:
            matches.append(field_match)
    if not matches:
        return 0.0
    matches.sort(reverse=True)
    return matches[0] + 0.75 * sum(matches[1:])


def candidate_updated_at(candidate: dict[str, Any]) -> date | None:
    return parse_optional_date(candidate.get("updated_at"))


def matches_structured_filters(candidate: dict[str, Any], plan: QueryPlan) -> bool:
    if plan.formats:
        candidate_format = canonical_format(str(candidate.get("format") or ""))
        if candidate_format not in plan.formats:
            return False

    updated_at = candidate_updated_at(candidate)
    if plan.updated_after and (updated_at is None or updated_at < plan.updated_after):
        return False
    if plan.updated_before and (updated_at is None or updated_at > plan.updated_before):
        return False
    return True


def field_aware_score(query: dict[str, Any], candidate: dict[str, Any], reference_date: date) -> float | None:
    plan = build_query_plan(query)
    if not matches_structured_filters(candidate, plan):
        return None

    lexical_score = field_aware_lexical_score(plan, candidate) if plan.subject else 0.0
    if plan.subject and lexical_score < MIN_LEXICAL_SCORE:
        return None
    has_structured_signal = bool(plan.formats or plan.recent or plan.updated_after or plan.updated_before)
    if not plan.subject and not has_structured_signal:
        return None

    score = lexical_score
    if plan.formats:
        score += 2.0
    if plan.updated_after or plan.updated_before:
        score += 1.0
    if plan.recent:
        updated_at = candidate_updated_at(candidate)
        if updated_at:
            age_days = max(0, (reference_date - updated_at).days)
            score += 3.0 / (1.0 + age_days / 7.0)
    return score


def baseline_target_text(candidate: dict[str, Any]) -> str:
    if candidate["kind"] == "contact":
        fields = ("name", "nickname", "organization", "phones", "emails")
    else:
        fields = ("title", "path", "summary", "format", "tags")
    return " ".join(item for field in fields for item in values(candidate, field)).casefold()


def baseline_query_terms(query: dict[str, Any]) -> set[str]:
    terms: set[str] = set()
    for raw in (query.get("keyword"), query.get("phrase")):
        if raw is None:
            continue
        text = str(raw).replace("/", " ").replace("_", " ")
        pieces = text.split()
        terms.update(pieces or [text])
    return {term for term in terms if term}


def baseline_score(query: dict[str, Any], candidate: dict[str, Any], _: date) -> float | None:
    haystack = baseline_target_text(candidate)
    score = 0.0
    for term in baseline_query_terms(query):
        if term.casefold() in haystack:
            score += 0.5 if len(term) <= 1 else 2.0

    qualifiers = query.get("qualifiers", {})
    for item in qualifiers.get("format", []):
        if str(item).casefold() in haystack:
            score += 1.0
    if "recent" in qualifiers.get("selection_hint", []):
        score += 0.25
    return score if score > 0 else None


def candidate_sort_title(candidate: dict[str, Any]) -> str:
    return str(candidate.get("title") or candidate.get("name") or candidate["id"])


ScoreFunction = Callable[[dict[str, Any], dict[str, Any], date], float | None]


def rank_candidates(
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    reference_date: date,
    score_function: ScoreFunction,
) -> list[tuple[str, float]]:
    scored: list[tuple[dict[str, Any], float]] = []
    for candidate in candidates:
        if candidate["kind"] != query["kind"]:
            continue
        score = score_function(query, candidate, reference_date)
        if score is not None:
            scored.append((candidate, score))
    scored.sort(key=lambda item: (-item[1], candidate_sort_title(item[0]), item[0]["id"]))
    return [(candidate["id"], score) for candidate, score in scored]


def empty_metric_bucket() -> dict[str, Any]:
    return {
        "closed_queries": 0,
        "top1_correct": 0,
        "reciprocal_rank_sum": 0.0,
        "recall_at_3_sum": 0.0,
        "open_queries": 0,
        "open_false_positives": 0,
    }


def finalize_metrics(bucket: dict[str, Any]) -> dict[str, Any]:
    closed = bucket["closed_queries"]
    opened = bucket["open_queries"]
    return {
        "closed_queries": closed,
        "top1": bucket["top1_correct"] / closed if closed else None,
        "mrr": bucket["reciprocal_rank_sum"] / closed if closed else None,
        "recall_at_3": bucket["recall_at_3_sum"] / closed if closed else None,
        "open_queries": opened,
        "open_false_positives": bucket["open_false_positives"],
        "open_false_positive_rate": bucket["open_false_positives"] / opened if opened else None,
    }


def evaluate_method(
    name: str,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    reference_date: date,
    score_function: ScoreFunction,
) -> dict[str, Any]:
    buckets = {kind: empty_metric_bucket() for kind in (*SUPPORTED_KINDS, "overall")}
    details: list[dict[str, Any]] = []

    for query in queries:
        ranked = rank_candidates(query, candidates, reference_date, score_function)
        retrieved = [candidate_id for candidate_id, _ in ranked]
        expected = list(query["expected"])
        query_buckets = (buckets[query["kind"]], buckets["overall"])

        if expected:
            expected_set = set(expected)
            first_relevant_rank = next(
                (index for index, candidate_id in enumerate(retrieved, start=1) if candidate_id in expected_set),
                None,
            )
            recall_at_3 = len(expected_set & set(retrieved[:3])) / len(expected_set)
            for bucket in query_buckets:
                bucket["closed_queries"] += 1
                bucket["top1_correct"] += bool(retrieved and retrieved[0] in expected_set)
                bucket["reciprocal_rank_sum"] += 1.0 / first_relevant_rank if first_relevant_rank else 0.0
                bucket["recall_at_3_sum"] += recall_at_3
        else:
            false_positive = bool(retrieved)
            for bucket in query_buckets:
                bucket["open_queries"] += 1
                bucket["open_false_positives"] += false_positive

        details.append(
            {
                "query_id": query["id"],
                "kind": query["kind"],
                "expected": expected,
                "top_results": [
                    {"id": candidate_id, "score": round(score, 6)} for candidate_id, score in ranked[:3]
                ],
            }
        )

    return {
        "method": name,
        "metrics": {kind: finalize_metrics(bucket) for kind, bucket in buckets.items()},
        "details": details,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    candidate_ids = [candidate["id"] for candidate in manifest["candidates"]]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")
    query_ids = [query["id"] for query in manifest["queries"]]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")

    candidates_by_id = {candidate["id"]: candidate for candidate in manifest["candidates"]}
    for candidate in manifest["candidates"]:
        if candidate["kind"] not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported candidate kind: {candidate['kind']}")
    for query in manifest["queries"]:
        if query["kind"] not in SUPPORTED_KINDS:
            raise ValueError(f"unsupported query kind: {query['kind']}")
        for candidate_id in query["expected"]:
            if candidate_id not in candidates_by_id:
                raise ValueError(f"query {query['id']} references missing candidate {candidate_id}")
            if candidates_by_id[candidate_id]["kind"] != query["kind"]:
                raise ValueError(f"query {query['id']} expected candidate has a different kind")

    for kind in SUPPORTED_KINDS:
        kind_queries = [query for query in manifest["queries"] if query["kind"] == kind]
        if not any(query["expected"] for query in kind_queries):
            raise ValueError(f"{kind} needs at least one closed-set query")
        if not any(not query["expected"] for query in kind_queries):
            raise ValueError(f"{kind} needs at least one open-set query")


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def print_summary(results: Iterable[dict[str, Any]]) -> None:
    print("method       kind     closed  top1   mrr    recall@3  open  open_fp  open_fp_rate")
    for result in results:
        for kind in (*SUPPORTED_KINDS, "overall"):
            metric = result["metrics"][kind]
            print(
                f"{result['method']:<12} {kind:<8} "
                f"{metric['closed_queries']:>6}  "
                f"{format_metric(metric['top1']):>5}  "
                f"{format_metric(metric['mrr']):>5}  "
                f"{format_metric(metric['recall_at_3']):>8}  "
                f"{metric['open_queries']:>4}  "
                f"{metric['open_false_positives']:>7}  "
                f"{format_metric(metric['open_false_positive_rate']):>12}"
            )


def print_failures(results: Iterable[dict[str, Any]]) -> None:
    for result in results:
        failures = []
        for detail in result["details"]:
            retrieved = [item["id"] for item in detail["top_results"]]
            expected = set(detail["expected"])
            failed = (expected and (not retrieved or retrieved[0] not in expected)) or (not expected and retrieved)
            if failed:
                failures.append(detail)
        print(f"\n{result['method']} failures ({len(failures)}):")
        for failure in failures:
            print(
                f"- {failure['query_id']}: expected={failure['expected']} "
                f"top_results={failure['top_results']}"
            )


def validate_regression_gate(results: list[dict[str, Any]]) -> None:
    proposed = next(result for result in results if result["method"] == "field_aware")
    for kind in (*SUPPORTED_KINDS, "overall"):
        metrics = proposed["metrics"][kind]
        if metrics["top1"] is None or metrics["top1"] < 0.90:
            raise SystemExit(f"field-aware {kind} Top-1 regressed below 0.90: {metrics['top1']}")
        if metrics["recall_at_3"] is None or metrics["recall_at_3"] < 0.90:
            raise SystemExit(
                f"field-aware {kind} Recall@3 regressed below 0.90: {metrics['recall_at_3']}"
            )
        if metrics["open_false_positives"] != 0:
            raise SystemExit(
                f"field-aware {kind} produced {metrics['open_false_positives']} open-set false positives"
            )


def manifest_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    parser.add_argument("--details", action="store_true", help="Print failed queries and their top results")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    reference_date = date.fromisoformat(manifest["reference_date"])
    methods = (
        ("baseline", baseline_score),
        ("field_aware", field_aware_score),
    )
    results = [
        evaluate_method(
            name,
            manifest["queries"],
            manifest["candidates"],
            reference_date,
            score_function,
        )
        for name, score_function in methods
    ]

    print_summary(results)
    validate_regression_gate(results)
    if args.details:
        print_failures(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": manifest_display_path(args.manifest),
            "reference_date": reference_date.isoformat(),
            "results": results,
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
