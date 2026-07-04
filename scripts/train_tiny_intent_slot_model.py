import argparse
import json
import random
import statistics
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

from env_guard import ensure_project_venv
from prepare_dataset import normalize_slots


ensure_project_venv()

SPAN_TAGS = ("B", "I")
SPAN_LABELS = ["O", "B", "I"]
TRANSFER_CUES = [
    "发给",
    "发送给",
    "传给",
    "传送给",
    "转给",
    "分享给",
    "丢给",
    "给",
    "发到",
    "发送到",
    "传到",
]
ACTION_CUES = [
    "发",
    "发送",
    "传",
    "传送",
    "转",
    "转发",
    "分享",
    "丢",
]


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


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


def char_type(ch):
    if "\u4e00" <= ch <= "\u9fff":
        return "cjk"
    if ch.isdigit():
        return "digit"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isspace():
        return "space"
    if ch in "_-（）()[]【】/\\.:：,，。;；":
        return "punct"
    return "other"


def length_bucket(text):
    length = len(text)
    if length <= 8:
        return "short"
    if length <= 16:
        return "medium"
    if length <= 28:
        return "long"
    return "very_long"


def add_feature(features, name, value=1.0):
    features[name] += value


def intent_features(text):
    features = Counter()
    add_feature(features, "bias")
    add_feature(features, f"len={length_bucket(text)}")

    for n in (1, 2, 3):
        if len(text) < n:
            continue
        for index in range(len(text) - n + 1):
            add_feature(features, f"ng{n}:{text[index:index + n]}")

    for prefix_len in (1, 2, 3):
        if len(text) >= prefix_len:
            add_feature(features, f"prefix{prefix_len}:{text[:prefix_len]}")
            add_feature(features, f"suffix{prefix_len}:{text[-prefix_len:]}")

    return dict(features)


def token_at(chars, index):
    if 0 <= index < len(chars):
        return chars[index]
    return "<BOS>" if index < 0 else "<EOS>"


def tag_features(chars, index, prev_tag):
    ch = token_at(chars, index)
    prev_ch = token_at(chars, index - 1)
    next_ch = token_at(chars, index + 1)
    prev2_ch = token_at(chars, index - 2)
    next2_ch = token_at(chars, index + 2)
    text = "".join(chars)
    left_context = text[max(0, index - 10) : index]
    right_context = text[index + 1 : index + 11]

    features = Counter()
    add_feature(features, "bias")
    add_feature(features, f"prev_tag={prev_tag}")
    add_feature(features, f"ch={ch}")
    add_feature(features, f"prev_ch={prev_ch}")
    add_feature(features, f"next_ch={next_ch}")
    add_feature(features, f"prev2_ch={prev2_ch}")
    add_feature(features, f"next2_ch={next2_ch}")
    add_feature(features, f"type={char_type(ch)}")
    add_feature(features, f"prev_type={char_type(prev_ch)}")
    add_feature(features, f"next_type={char_type(next_ch)}")
    add_feature(features, f"bigram_prev={prev_ch}{ch}")
    add_feature(features, f"bigram_next={ch}{next_ch}")
    add_feature(features, f"trigram={prev_ch}{ch}{next_ch}")

    for size in range(1, 6):
        left = text[max(0, index - size) : index]
        right = text[index + 1 : index + 1 + size]
        current_prefix = text[index : index + size]
        current_suffix = text[max(0, index - size + 1) : index + 1]
        add_feature(features, f"left{size}={left}")
        add_feature(features, f"right{size}={right}")
        add_feature(features, f"prefix_at{size}={current_prefix}")
        add_feature(features, f"suffix_at{size}={current_suffix}")

    for cue in TRANSFER_CUES:
        if left_context.endswith(cue):
            add_feature(features, f"left_endswith_cue={cue}")
        if cue in left_context:
            add_feature(features, f"left_has_cue={cue}")
        if right_context.startswith(cue):
            add_feature(features, f"right_startswith_cue={cue}")
        if cue in right_context:
            add_feature(features, f"right_has_cue={cue}")

    for cue in ACTION_CUES:
        if left_context.endswith(cue):
            add_feature(features, f"left_endswith_action={cue}")
        if cue in left_context:
            add_feature(features, f"left_has_action={cue}")
        if right_context.startswith(cue):
            add_feature(features, f"right_startswith_action={cue}")
        if cue in right_context:
            add_feature(features, f"right_has_action={cue}")

    if "给" in left_context and any(cue in right_context for cue in ACTION_CUES):
        add_feature(features, "between_give_and_action")

    if index == 0:
        add_feature(features, "is_start")
    if index == len(chars) - 1:
        add_feature(features, "is_end")

    relative_pos = int(10 * index / max(1, len(chars)))
    add_feature(features, f"pos_bucket={relative_pos}")
    return dict(features)


class LinearClassifier:
    def __init__(self, labels, weights=None):
        self.labels = list(labels)
        self.weights = defaultdict(dict)
        if weights:
            for feature, label_weights in weights.items():
                self.weights[feature] = dict(label_weights)

    def scores(self, features):
        scores = {label: 0.0 for label in self.labels}
        for feature, value in features.items():
            label_weights = self.weights.get(feature)
            if not label_weights:
                continue
            for label, weight in label_weights.items():
                scores[label] += weight * value
        return scores

    def predict(self, features):
        scores = self.scores(features)
        return max(self.labels, key=lambda label: (scores[label], label))

    def update(self, truth, guess, features):
        if truth == guess:
            return False
        for feature, value in features.items():
            label_weights = self.weights[feature]
            label_weights[truth] = label_weights.get(truth, 0.0) + value
            label_weights[guess] = label_weights.get(guess, 0.0) - value
        return True

    def prune(self):
        pruned = {}
        for feature, label_weights in self.weights.items():
            kept = {
                label: weight
                for label, weight in label_weights.items()
                if abs(weight) > 1e-12
            }
            if kept:
                pruned[feature] = kept
        self.weights = defaultdict(dict, pruned)

    def to_payload(self):
        self.prune()
        return {
            "labels": self.labels,
            "weights": dict(sorted(self.weights.items())),
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(payload["labels"], payload["weights"])


def find_occurrences(text, phrase):
    starts = []
    start = text.find(phrase)
    while start >= 0:
        starts.append(start)
        start = text.find(phrase, start + 1)
    return starts


def spans_overlap(left_start, left_end, right_start, right_end):
    return not (left_end <= right_start or right_end <= left_start)


def choose_slot_span(row, slot_key):
    text = row["input"]
    phrase = row["raw_slots"][slot_key]
    starts = find_occurrences(text, phrase)
    if not starts:
        raise ValueError(f"slot phrase not found in input: {phrase!r} / {text!r}")

    if slot_key == "share_target" and "share_content" in row["raw_slots"]:
        content = row["raw_slots"]["share_content"]
        content_start = find_occurrences(text, content)[0]
        content_end = content_start + len(content)
        non_overlapping = [
            start
            for start in starts
            if not spans_overlap(
                start, start + len(phrase), content_start, content_end
            )
        ]
        if non_overlapping:
            after_content = [start for start in non_overlapping if start >= content_end]
            start = after_content[0] if after_content else non_overlapping[0]
            return start, start + len(phrase)

    start = starts[0]
    return start, start + len(phrase)


def apply_span(tags, start, end):
    for index in range(start, end):
        tags[index] = "B" if index == start else "I"


def gold_span_tags(row, slot_key):
    text = row["input"]
    tags = ["O"] * len(text)
    if row["intent"] == "unknown":
        return tags

    start, end = choose_slot_span(row, slot_key)
    apply_span(tags, start, end)
    return tags


def normalize_tag_transition(tag, prev_tag):
    if tag == "I" and prev_tag not in SPAN_TAGS:
        return "B"
    return tag


def span_tags_to_text(text, tags):
    index = 0
    while index < len(tags):
        tag = tags[index]
        if tag == "B":
            start = index
            index += 1
            while index < len(tags) and tags[index] == "I":
                index += 1
            return text[start:index]
        index += 1
    return None


class TinyIntentSlotModel:
    def __init__(self, intent_model, content_model, target_model, metadata=None):
        self.intent_model = intent_model
        self.content_model = content_model
        self.target_model = target_model
        self.metadata = metadata or {}

    def predict_span_tags(self, text, span_model):
        chars = list(text)
        tags = []
        prev_tag = "<START>"
        for index in range(len(chars)):
            features = tag_features(chars, index, prev_tag)
            tag = span_model.predict(features)
            tag = normalize_tag_transition(tag, prev_tag)
            tags.append(tag)
            prev_tag = tag
        return tags

    def predict(self, text):
        intent = self.intent_model.predict(intent_features(text))
        content_tags = self.predict_span_tags(text, self.content_model)
        target_tags = self.predict_span_tags(text, self.target_model)
        if intent == "unknown":
            slots = {}
        else:
            slots = {}
            content = span_tags_to_text(text, content_tags)
            target = span_tags_to_text(text, target_tags)
            if content is not None:
                slots["share_content"] = content
            if target is not None:
                slots["share_target"] = target
        return {
            "intent": intent,
            "slots": slots,
            "content_tags": content_tags,
            "target_tags": target_tags,
        }

    def to_payload(self):
        return {
            "version": 1,
            "metadata": self.metadata,
            "intent_model": self.intent_model.to_payload(),
            "content_model": self.content_model.to_payload(),
            "target_model": self.target_model.to_payload(),
        }

    @classmethod
    def from_payload(cls, payload):
        return cls(
            LinearClassifier.from_payload(payload["intent_model"]),
            LinearClassifier.from_payload(payload["content_model"]),
            LinearClassifier.from_payload(payload["target_model"]),
            payload.get("metadata", {}),
        )

    @classmethod
    def load(cls, path):
        with path.open("r", encoding="utf-8") as f:
            return cls.from_payload(json.load(f))


def train_intent_model(rows, epochs, seed):
    labels = sorted({row["intent"] for row in rows})
    model = LinearClassifier(labels)
    rng = random.Random(seed)
    training_rows = list(rows)

    for _epoch in range(epochs):
        rng.shuffle(training_rows)
        for row in training_rows:
            features = intent_features(row["input"])
            guess = model.predict(features)
            model.update(row["intent"], guess, features)

    model.prune()
    return model


def train_span_model(rows, slot_key, epochs, seed):
    model = LinearClassifier(SPAN_LABELS)
    rng = random.Random(seed)
    training_rows = list(rows)

    for _epoch in range(epochs):
        rng.shuffle(training_rows)
        for row in training_rows:
            chars = list(row["input"])
            tags = gold_span_tags(row, slot_key)
            prev_tag = "<START>"
            for index, truth in enumerate(tags):
                features = tag_features(chars, index, prev_tag)
                guess = model.predict(features)
                model.update(truth, guess, features)
                prev_tag = truth

    model.prune()
    return model


def normalize_predicted_slots(intent, slots):
    if intent == "unknown":
        return {}
    if "share_content" not in slots or "share_target" not in slots:
        return {}
    return normalize_slots(intent, slots)


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


def evaluate(model, rows):
    counts = Counter()
    per_intent = defaultdict(Counter)
    latencies_ms = []

    for row in rows:
        start = time.perf_counter()
        prediction = model.predict(row["input"])
        latencies_ms.append((time.perf_counter() - start) * 1000)

        true_intent = row["intent"]
        true_slots = row["raw_slots"]
        pred_intent = prediction["intent"]
        pred_slots = prediction["slots"]
        pred_normalized = normalize_predicted_slots(pred_intent, pred_slots)

        counts["rows"] += 1
        counts["intent_correct"] += int(pred_intent == true_intent)
        counts["slot_frame_exact"] += int(pred_slots == true_slots)
        counts["complete_output_exact"] += int(
            pred_intent == true_intent and pred_slots == true_slots
        )
        counts["normalized_exact"] += int(pred_normalized == row["normalized_slots"])

        intent_bucket = per_intent[true_intent]
        intent_bucket["rows"] += 1
        intent_bucket["intent_correct"] += int(pred_intent == true_intent)
        intent_bucket["slot_frame_exact"] += int(pred_slots == true_slots)
        intent_bucket["complete_output_exact"] += int(
            pred_intent == true_intent and pred_slots == true_slots
        )
        intent_bucket["normalized_exact"] += int(
            pred_normalized == row["normalized_slots"]
        )

        if true_intent != "unknown":
            counts["non_unknown_rows"] += 1
            content_ok = pred_slots.get("share_content") == true_slots["share_content"]
            target_ok = pred_slots.get("share_target") == true_slots["share_target"]
            keyword_ok = (
                pred_normalized.get("search_keyword")
                == row["normalized_slots"].get("search_keyword")
            )
            counts["share_content_exact"] += int(content_ok)
            counts["share_target_exact"] += int(target_ok)
            counts["search_keyword_exact"] += int(keyword_ok)

            intent_bucket["non_unknown_rows"] += 1
            intent_bucket["share_content_exact"] += int(content_ok)
            intent_bucket["share_target_exact"] += int(target_ok)
            intent_bucket["search_keyword_exact"] += int(keyword_ok)

    def rate(numerator, denominator):
        return round(numerator / denominator, 6) if denominator else 0.0

    metrics = {
        "rows": counts["rows"],
        "intent_accuracy": rate(counts["intent_correct"], counts["rows"]),
        "slot_frame_exact": rate(counts["slot_frame_exact"], counts["rows"]),
        "normalized_exact": rate(counts["normalized_exact"], counts["rows"]),
        "complete_output_exact": rate(counts["complete_output_exact"], counts["rows"]),
        "share_content_exact": rate(
            counts["share_content_exact"], counts["non_unknown_rows"]
        ),
        "share_target_exact": rate(
            counts["share_target_exact"], counts["non_unknown_rows"]
        ),
        "search_keyword_exact": rate(
            counts["search_keyword_exact"], counts["non_unknown_rows"]
        ),
        "latency_ms_avg": round(statistics.mean(latencies_ms), 6)
        if latencies_ms
        else 0.0,
        "latency_ms_p95": round(percentile(latencies_ms, 95), 6),
    }

    by_intent = {}
    for intent, bucket in sorted(per_intent.items()):
        by_intent[intent] = {
            "rows": bucket["rows"],
            "intent_accuracy": rate(bucket["intent_correct"], bucket["rows"]),
            "slot_frame_exact": rate(bucket["slot_frame_exact"], bucket["rows"]),
            "normalized_exact": rate(bucket["normalized_exact"], bucket["rows"]),
            "complete_output_exact": rate(
                bucket["complete_output_exact"], bucket["rows"]
            ),
            "share_content_exact": rate(
                bucket["share_content_exact"], bucket["non_unknown_rows"]
            ),
            "share_target_exact": rate(
                bucket["share_target_exact"], bucket["non_unknown_rows"]
            ),
            "search_keyword_exact": rate(
                bucket["search_keyword_exact"], bucket["non_unknown_rows"]
            ),
        }
    metrics["by_intent"] = by_intent
    return metrics


def collect_failures(model, rows, limit=20):
    failures = []
    for row in rows:
        prediction = model.predict(row["input"])
        if (
            prediction["intent"] != row["intent"]
            or prediction["slots"] != row["raw_slots"]
        ):
            failures.append(
                {
                    "input": row["input"],
                    "true": {
                        "intent": row["intent"],
                        "slots": row["raw_slots"],
                    },
                    "pred": {
                        "intent": prediction["intent"],
                        "slots": prediction["slots"],
                    },
                }
            )
            if len(failures) >= limit:
                break
    return failures


def measure_load(model_path):
    tracemalloc.start()
    start = time.perf_counter()
    TinyIntentSlotModel.load(model_path)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "load_time_ms": round(elapsed_ms, 6),
        "peak_load_memory_kb": round(peak / 1024, 3),
    }


def render_report(payload):
    dataset_rows = []
    for name, metrics in payload["evaluation"].items():
        dataset_rows.append(
            [
                name,
                metrics["rows"],
                metrics["intent_accuracy"],
                metrics["share_content_exact"],
                metrics["share_target_exact"],
                metrics["search_keyword_exact"],
                metrics["complete_output_exact"],
                metrics["latency_ms_avg"],
                metrics["latency_ms_p95"],
            ]
        )

    lines = [
        "# 第二阶段小模型训练报告",
        "",
        "## 模型",
        "",
        "- intent：字符 1-3 gram 线性分类器。",
        "- slots：两个字符级 BIO span 标注器，分别抽取 `share_content` 和 `share_target`，可处理重叠槽位。",
        "- 训练和评估均使用项目 `.venv`，脚本只依赖 Python 标准库。",
        "- 导出格式为 JSON 权重，后续可以在 Swift 端复现同样特征和贪心解码。",
        "",
        "## 训练配置",
        "",
        "```json",
        json.dumps(payload["training"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 总体指标",
        "",
        render_table(
            [
                "dataset",
                "rows",
                "intent_acc",
                "content_exact",
                "target_exact",
                "keyword_exact",
                "complete_exact",
                "avg_ms",
                "p95_ms",
            ],
            dataset_rows,
        ),
        "",
        "## 模型体积与加载",
        "",
        "```json",
        json.dumps(payload["artifact"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 失败样例",
        "",
        "```json",
        json.dumps(payload["failures"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 说明",
        "",
        "- `complete_output_exact` 要求 intent、`share_content`、`share_target` 全部完全匹配。",
        "- `search_keyword_exact` 先用第一阶段相同标准化逻辑生成检索关键词，再和标注侧对齐。",
        "- Python 延迟仅作为开发机基线；最终指标仍需要在 iPhone 真机侧用 App 统计。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Train a tiny intent and slot model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("processed") / "datasets",
        help="Directory containing normalized jsonl datasets.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models") / "tiny_intent_slot_model.json",
        help="Output JSON model path.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("reports") / "phase2_model_metrics.json",
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports") / "phase2_model_report.md",
        help="Output Markdown report path.",
    )
    parser.add_argument("--intent-epochs", type=int, default=10)
    parser.add_argument("--tag-epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    dataset_files = {
        "positive_train": "positive_train.jsonl",
        "positive_valid": "positive_valid.jsonl",
        "positive_cold_start_test": "positive_cold_start_test.jsonl",
        "with_unknown_train": "with_unknown_train.jsonl",
        "with_unknown_valid": "with_unknown_valid.jsonl",
        "with_unknown_cold_start_test": "with_unknown_cold_start_test.jsonl",
    }
    datasets = {
        name: load_jsonl(args.data_dir / file_name)
        for name, file_name in dataset_files.items()
    }

    train_rows = datasets["positive_train"] + datasets["with_unknown_train"]
    start_train = time.perf_counter()
    intent_model = train_intent_model(train_rows, args.intent_epochs, args.seed)
    content_model = train_span_model(
        train_rows, "share_content", args.tag_epochs, args.seed + 1
    )
    target_model = train_span_model(
        train_rows, "share_target", args.tag_epochs, args.seed + 2
    )
    train_time_ms = (time.perf_counter() - start_train) * 1000

    model = TinyIntentSlotModel(
        intent_model,
        content_model,
        target_model,
        metadata={
            "created_by": "scripts/train_tiny_intent_slot_model.py",
            "model_family": "char_ngram_linear_intent_plus_two_char_bio_span_extractors",
            "train_datasets": ["positive_train", "with_unknown_train"],
            "intent_epochs": args.intent_epochs,
            "tag_epochs": args.tag_epochs,
            "seed": args.seed,
        },
    )
    dump_json(args.model_out, model.to_payload())

    evaluation = {
        name: evaluate(model, rows)
        for name, rows in datasets.items()
    }
    artifact = {
        "model_path": str(args.model_out),
        "model_size_bytes": args.model_out.stat().st_size,
        "model_size_mb": round(args.model_out.stat().st_size / (1024 * 1024), 6),
        "train_time_ms": round(train_time_ms, 6),
        **measure_load(args.model_out),
    }
    payload = {
        "training": {
            "train_rows": len(train_rows),
            "intent_epochs": args.intent_epochs,
            "tag_epochs": args.tag_epochs,
            "seed": args.seed,
            "datasets": {name: len(rows) for name, rows in datasets.items()},
        },
        "artifact": artifact,
        "evaluation": evaluation,
        "failures": {
            "with_unknown_valid": collect_failures(
                model, datasets["with_unknown_valid"], limit=20
            ),
            "with_unknown_cold_start_test": collect_failures(
                model, datasets["with_unknown_cold_start_test"], limit=20
            ),
        },
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(render_report(payload), encoding="utf-8")

    print(f"Wrote model: {args.model_out}")
    print(f"Wrote metrics: {args.metrics_out}")
    print(f"Wrote report: {args.report_out}")


if __name__ == "__main__":
    main()
