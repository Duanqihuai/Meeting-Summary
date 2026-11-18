"""透明的基础评测：CER/WER、ROUGE-L F1、标注重叠率；不依赖在线模型评分。"""
import argparse
import re
import unicodedata
from pathlib import Path
from common.core import read_json, write_json


def tokens(text, unit):
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return re.findall(r"\S+", text) if unit == "word" else [c for c in text if not c.isspace()]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = row
    return prev[-1]


def rouge_l(a, b):
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        row = [0]
        for j, y in enumerate(b, 1):
            row.append(prev[j - 1] + 1 if x == y else max(row[-1], prev[j]))
        prev = row
    return 2 * prev[-1] / (len(a) + len(b))


def overlap_ratio(segments):
    # 计算两个及以上不同说话人同时发言 / 至少一人发言的时间。
    # 同一说话人的多个重叠区间不会被误算成多说话人重叠。
    events = []
    for s in segments:
        if s.get("speaker") is None or s.get("start") is None:
            raise ValueError("重叠率需要带真实说话人和时间戳的参考标注")
        events += [(s["start"], 1, s["speaker"]), (s["end"], -1, s["speaker"])]
    active, last, speech, overlap = {}, 0, 0, 0
    for t, delta, who in sorted(events):
        n = sum(v > 0 for v in active.values())
        speech += (t - last) * (n >= 1)
        overlap += (t - last) * (n >= 2)
        active[who] = active.get(who, 0) + delta
        last = t
    return overlap / speech if speech else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="UTF-8 参考纯文本")
    p.add_argument("--prediction", required=True, help="UTF-8 预测纯文本")
    p.add_argument("--task", choices=["asr", "summary"], required=True)
    p.add_argument("--unit", choices=["char", "word"], default="char")
    p.add_argument("--reference-segments", help="可选，用参考标注计算重叠率")
    p.add_argument("--out", default="outputs/metrics.json")
    args = p.parse_args()
    ref = tokens(Path(args.reference).read_text(encoding="utf-8-sig"), args.unit)
    hyp = tokens(Path(args.prediction).read_text(encoding="utf-8-sig"), args.unit)
    if not ref:
        raise ValueError("参考文本为空，不能评测")
    result = {"task": args.task, "unit": args.unit, "reference_tokens": len(ref)}
    if args.task == "asr":
        errors = edit_distance(ref, hyp)
        result.update(errors=errors, error_rate=errors / len(ref))
    else:
        result["rouge_l_f1"] = rouge_l(ref, hyp)
    if args.reference_segments:
        result["overlap_ratio"] = overlap_ratio(read_json(args.reference_segments))
    write_json(args.out, result)
    print(result)


if __name__ == "__main__":
    main()
