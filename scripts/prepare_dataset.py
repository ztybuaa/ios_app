import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from env_guard import ensure_project_venv


ensure_project_venv()

DATASETS = [
    {
        "name": "positive_train",
        "suite": "positive_only",
        "split": "train",
        "file": "train_intent_slots(1).json",
    },
    {
        "name": "positive_valid",
        "suite": "positive_only",
        "split": "valid",
        "file": "valid_intent_slots(1).json",
    },
    {
        "name": "positive_cold_start_test",
        "suite": "positive_only",
        "split": "cold_start_test",
        "file": "cold_start_test_intent_slots(1).json",
    },
    {
        "name": "with_unknown_train",
        "suite": "with_unknown",
        "split": "train",
        "file": "train_intent_slots_with_unknown.json",
    },
    {
        "name": "with_unknown_valid",
        "suite": "with_unknown",
        "split": "valid",
        "file": "valid_intent_slots_with_unknown(1).json",
    },
    {
        "name": "with_unknown_cold_start_test",
        "suite": "with_unknown",
        "split": "cold_start_test",
        "file": "cold_start_test_intent_slots_with_unknown.json",
    },
]

RESOURCE_TYPES = {
    "photo": "图片/照片",
    "video": "视频",
    "file": "文件",
    "folder": "文件夹/目录",
    "contact": "联系人/联系方式",
}

RESOURCE_TERMS = {
    "photo": [
        "图片",
        "照片",
        "相片",
        "截图",
        "截屏",
        "风景照",
        "自拍",
        "动图",
        "GIF动图",
        "图",
        "相册",
    ],
    "video": [
        "视频",
        "录像",
        "录屏",
        "短视频",
        "短片",
        "影片",
        "MOV文件",
        "MP4文件",
        "MOV视频",
        "MP4视频",
        "MP4",
        "MOV",
    ],
    "file": [
        "文件",
        "文档",
        "表格",
        "合同",
        "报告",
        "PPT文件",
        "Word文件",
        "Excel表格",
        "PDF文件",
        "PDF文档",
        "Word文档",
        "PPT",
        "PDF",
        "Word",
        "Excel",
        "压缩包",
    ],
    "folder": [
        "文件夹",
        "资料夹",
        "目录",
    ],
    "contact": [
        "联系方式",
        "联系人信息",
        "联系人",
        "通讯录",
        "名片",
        "电子名片",
        "VCF名片",
        "vCard联系人",
        "手机号",
        "电话号码",
        "电话",
        "微信号",
        "邮箱地址",
        "办公地址",
        "地址",
        "二维码",
    ],
}

SELECTION_HINT_PATTERNS = {
    "current_selection": [
        "这张",
        "这个",
        "这份",
        "这段",
        "这个",
        "这几张",
        "选中的",
        "当前",
        "刚发现的",
    ],
    "recent": [
        "最近",
        "刚才",
        "最新",
        "新拍",
        "刚保存",
        "下载好",
        "下载的",
        "上次",
        "今天新增",
    ],
    "all": [
        "全部",
        "所有",
        "多个",
        "部分",
        "十个",
        "五个",
        "三张",
    ],
}

TIME_TERMS = [
    "今天",
    "昨天",
    "前天",
    "上周",
    "上个月",
    "最近",
    "刚才",
    "最新",
    "近期",
    "去年",
    "今年",
    "明天",
]

FORMAT_TERMS = [
    "PDF",
    "Word",
    "Excel",
    "PPT",
    "PNG",
    "JPG",
    "GIF",
    "MP4",
    "MOV",
    "VCF",
    "vCard",
]

GENERIC_PREFIXES = [
    "这个",
    "那个",
    "这张",
    "那张",
    "这份",
    "那份",
    "这段",
    "那段",
    "这些",
    "那些",
    "部分",
    "多个",
    "几个",
    "全部",
    "所有",
    "当前",
    "选中的",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def parse_model_output(raw_output, source_file, row_index):
    try:
        output = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{source_file} row {row_index}: output is not valid JSON: {exc}"
        ) from exc

    if not isinstance(output, dict):
        raise ValueError(f"{source_file} row {row_index}: output must be an object")
    if "intent" not in output:
        raise ValueError(f"{source_file} row {row_index}: missing intent")

    slots = output.get("slots")
    if slots is None:
        slots = {}
    if not isinstance(slots, dict):
        raise ValueError(f"{source_file} row {row_index}: slots must be an object")

    return {
        "intent": output["intent"],
        "slots": slots,
    }


def remove_terms(text, terms):
    ordered_terms = sorted(terms, key=len, reverse=True)
    result = text
    for term in ordered_terms:
        result = result.replace(term, "")
    return result


def strip_generic_markers(text):
    result = text
    changed = True
    while changed:
        changed = False
        for prefix in GENERIC_PREFIXES:
            if result.startswith(prefix):
                result = result[len(prefix) :]
                changed = True
        for suffix in ["一下", "过去", "过来", "哈"]:
            if result.endswith(suffix):
                result = result[: -len(suffix)]
                changed = True
    return result


def compact_text(text):
    return re.sub(r"\s+", "", text).strip(" 的。、，,.;；：:")


def extract_selection_hints(text):
    hints = []
    for hint, patterns in SELECTION_HINT_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            hints.append(hint)
    return hints


def extract_qualifiers(text):
    return {
        "time": [term for term in TIME_TERMS if term in text],
        "format": [term for term in FORMAT_TERMS if term in text],
        "selection_hint": extract_selection_hints(text),
    }


def normalize_search_keyword(intent, resource_phrase):
    if intent == "unknown":
        return None

    terms = RESOURCE_TERMS.get(intent, [])
    keyword = remove_terms(resource_phrase, terms)
    keyword = strip_generic_markers(keyword)
    keyword = compact_text(keyword)

    # A blank keyword means the phrase is generic selection/context only, such as
    # "这张照片" or "这个文件". Keep it explicit instead of inventing a keyword.
    if not keyword:
        return None

    if keyword in {"的", "我", "我的", "这", "那"}:
        return None

    return keyword


def normalize_slots(intent, raw_slots):
    if intent == "unknown":
        return {}

    missing = [key for key in ("share_content", "share_target") if key not in raw_slots]
    if missing:
        raise ValueError(f"non-unknown intent {intent} is missing slots: {missing}")

    resource_phrase = str(raw_slots["share_content"])
    target = str(raw_slots["share_target"])

    return {
        "resource_type": RESOURCE_TYPES.get(intent, intent),
        "resource_phrase": resource_phrase,
        "search_keyword": normalize_search_keyword(intent, resource_phrase),
        "target": target,
        "target_keyword": target,
        "qualifiers": extract_qualifiers(resource_phrase),
    }


def build_record(dataset_meta, row_index, example):
    parsed_output = parse_model_output(example["output"], dataset_meta["file"], row_index)
    intent = parsed_output["intent"]
    raw_slots = parsed_output["slots"]

    if intent == "unknown" and raw_slots:
        raise ValueError(f"{dataset_meta['file']} row {row_index}: unknown has non-empty slots")

    normalized_slots = normalize_slots(intent, raw_slots)

    return {
        "id": f"{dataset_meta['name']}_{row_index:06d}",
        "suite": dataset_meta["suite"],
        "split": dataset_meta["split"],
        "source_file": dataset_meta["file"],
        "instruction": example.get("instruction", ""),
        "input": example.get("input", ""),
        "intent": intent,
        "raw_output": parsed_output,
        "raw_slots": raw_slots,
        "normalized_slots": normalized_slots,
    }


def summarize_records(records_by_dataset):
    summary = {
        "datasets": [],
        "overall": {
            "rows": 0,
            "intent_counts": Counter(),
            "slot_key_counts": Counter(),
            "search_keyword_counts": Counter(),
            "generic_resource_phrase_counts": Counter(),
        },
    }

    for meta, records in records_by_dataset:
        intent_counts = Counter(row["intent"] for row in records)
        slot_key_counts = Counter()
        search_keyword_counts = Counter()
        generic_resource_phrase_counts = Counter()
        examples = {}

        for row in records:
            intent = row["intent"]
            examples.setdefault(intent, row["input"])
            for key in row["raw_slots"]:
                slot_key_counts[key] += 1
            if intent != "unknown":
                if row["normalized_slots"].get("search_keyword"):
                    search_keyword_counts[intent] += 1
                else:
                    generic_resource_phrase_counts[intent] += 1

        dataset_summary = {
            "name": meta["name"],
            "suite": meta["suite"],
            "split": meta["split"],
            "source_file": meta["file"],
            "rows": len(records),
            "intent_counts": dict(intent_counts),
            "slot_key_counts": dict(slot_key_counts),
            "search_keyword_counts": dict(search_keyword_counts),
            "generic_resource_phrase_counts": dict(generic_resource_phrase_counts),
            "examples": examples,
        }
        summary["datasets"].append(dataset_summary)
        summary["overall"]["rows"] += len(records)
        summary["overall"]["intent_counts"].update(intent_counts)
        summary["overall"]["slot_key_counts"].update(slot_key_counts)
        summary["overall"]["search_keyword_counts"].update(search_keyword_counts)
        summary["overall"]["generic_resource_phrase_counts"].update(
            generic_resource_phrase_counts
        )

    summary["overall"]["intent_counts"] = dict(summary["overall"]["intent_counts"])
    summary["overall"]["slot_key_counts"] = dict(summary["overall"]["slot_key_counts"])
    summary["overall"]["search_keyword_counts"] = dict(
        summary["overall"]["search_keyword_counts"]
    )
    summary["overall"]["generic_resource_phrase_counts"] = dict(
        summary["overall"]["generic_resource_phrase_counts"]
    )
    return summary


def top_share_content(records, limit=12):
    by_intent = defaultdict(Counter)
    for row in records:
        if row["intent"] == "unknown":
            continue
        by_intent[row["intent"]][row["raw_slots"]["share_content"]] += 1
    return {
        intent: counter.most_common(limit)
        for intent, counter in sorted(by_intent.items())
    }


def render_table(headers, rows):
    widths = [len(header) for header in headers]
    normalized_rows = []
    for row in rows:
        normalized = [str(cell) for cell in row]
        normalized_rows.append(normalized)
        widths = [max(width, len(cell)) for width, cell in zip(widths, normalized)]

    header_line = "| " + " | ".join(
        header.ljust(width) for header, width in zip(headers, widths)
    ) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body_lines = [
        "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"
        for row in normalized_rows
    ]
    return "\n".join([header_line, sep_line] + body_lines)


def render_report(summary, all_records, top_content):
    dataset_rows = []
    for item in summary["datasets"]:
        intent_text = ", ".join(
            f"{intent}:{count}" for intent, count in sorted(item["intent_counts"].items())
        )
        keyword_text = ", ".join(
            f"{intent}:{count}"
            for intent, count in sorted(item["search_keyword_counts"].items())
        )
        generic_text = ", ".join(
            f"{intent}:{count}"
            for intent, count in sorted(item["generic_resource_phrase_counts"].items())
        )
        dataset_rows.append(
            [
                item["source_file"],
                item["suite"],
                item["split"],
                item["rows"],
                intent_text,
                keyword_text or "-",
                generic_text or "-",
            ]
        )

    example_rows = []
    seen_intents = set()
    for row in all_records:
        intent = row["intent"]
        if intent in seen_intents:
            continue
        seen_intents.add(intent)
        example_rows.append(
            [
                intent,
                row["input"],
                json.dumps(row["normalized_slots"], ensure_ascii=False),
            ]
        )

    top_sections = []
    for intent, rows in top_content.items():
        top_sections.append(f"### {intent}")
        top_sections.append(render_table(["share_content", "count"], rows))
        top_sections.append("")

    lines = [
        "# 第一阶段数据整理报告",
        "",
        "## 结论",
        "",
        "- 六份原始 JSON 均已成功解析，`output` 字段均为合法 JSON 字符串。",
        "- 原始监督信号包含 `intent`、`share_content`、`share_target`；`unknown` 样本的 `slots` 为空。",
        "- 标准化数据额外生成 `normalized_slots`，保留 `resource_phrase`，并派生 `search_keyword`、`target`、`qualifiers`。",
        "- 对 `这张照片`、`这个文件` 这类泛指资源，`search_keyword` 显式置为 `null`，依赖 `selection_hint` 或上下文资源选择，不虚构检索关键词。",
        "",
        "## 数据集统计",
        "",
        render_table(
            [
                "file",
                "suite",
                "split",
                "rows",
                "intent_counts",
                "keyword_rows",
                "generic_rows",
            ],
            dataset_rows,
        ),
        "",
        "## 总体统计",
        "",
        "```json",
        json.dumps(summary["overall"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 标准化槽位示例",
        "",
        render_table(["intent", "input", "normalized_slots"], example_rows),
        "",
        "## 高频资源短语",
        "",
        *top_sections,
        "## 转换逻辑",
        "",
        "- `resource_phrase` 直接来自原始 `share_content`，用于训练和可解释展示。",
        "- `search_keyword` 从 `resource_phrase` 中去除 intent 对应资源类型词，例如 `图片/照片/文件夹/联系方式` 等；如果去除后没有有效语义词，则置为 `null`。",
        "- `target` 和 `target_keyword` 直接来自原始 `share_target`，供联系人或设备候选检索使用。",
        "- `qualifiers.time` 抽取 `今天/昨天/最近/刚才/上周/上个月` 等时间限定词。",
        "- `qualifiers.format` 抽取 `PDF/Word/Excel/PPT/PNG/JPG/GIF/MP4/MOV/VCF/vCard` 等格式限定词。",
        "- `qualifiers.selection_hint` 标记 `current_selection`、`recent`、`all` 等资源选择语义。",
        "",
        "## 输出文件",
        "",
        "- `processed/dataset_manifest.json`：六份数据的 manifest 和统计。",
        "- `processed/datasets/*.jsonl`：每份原始数据对应一份标准化 jsonl。",
        "- `processed/datasets/all.jsonl`：六份数据合并后的标准化 jsonl。",
        "",
        "## 下一阶段建议",
        "",
        "1. 用 `with_unknown` 套件训练完整的六类 intent + 槽位抽取模型。",
        "2. 用 `positive_only` 套件单独报告五类正向能力，避免 unknown 指标掩盖资源类表现。",
        "3. 将 `share_content/share_target` 转成字符级 BIO 标签，训练端侧可部署的小型槽位抽取模型。",
        "4. 先实现 `unknown` 拦截和 `contact` 检索模块，再接 photo/file/folder/video 模块。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Prepare intent-slot datasets for the demo app.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the six raw JSON dataset files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("processed"),
        help="Output directory for normalized data and manifest.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports") / "phase1_data_report.md",
        help="Markdown report path.",
    )
    args = parser.parse_args()

    records_by_dataset = []
    all_records = []

    for meta in DATASETS:
        source_path = args.root / meta["file"]
        examples = load_json(source_path)
        if not isinstance(examples, list):
            raise ValueError(f"{meta['file']}: dataset root must be a list")

        records = [build_record(meta, index, example) for index, example in enumerate(examples)]
        records_by_dataset.append((meta, records))
        all_records.extend(records)
        dump_jsonl(args.out / "datasets" / f"{meta['name']}.jsonl", records)

    dump_jsonl(args.out / "datasets" / "all.jsonl", all_records)

    summary = summarize_records(records_by_dataset)
    manifest = {
        "source_root": str(args.root.resolve()),
        "record_count": len(all_records),
        "datasets": summary["datasets"],
        "overall": summary["overall"],
        "normalized_schema": {
            "intent": "photo | video | file | folder | contact | unknown",
            "raw_slots": "original parsed slots from output JSON",
            "normalized_slots": {
                "resource_type": "canonical resource class label",
                "resource_phrase": "original share_content",
                "search_keyword": "resource retrieval keyword, null for generic/current-selection phrases",
                "target": "original share_target",
                "target_keyword": "target candidate retrieval keyword",
                "qualifiers": "time, format, and selection hints extracted from resource_phrase",
            },
        },
    }
    dump_json(args.out / "dataset_manifest.json", manifest)

    report = render_report(summary, all_records, top_share_content(all_records))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print(f"Wrote {len(all_records)} normalized records")
    print(f"Wrote manifest: {args.out / 'dataset_manifest.json'}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
