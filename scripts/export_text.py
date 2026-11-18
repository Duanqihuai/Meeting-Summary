"""从系统 JSON 提取评测纯文本，去除标题、证据标签和元数据。"""
import argparse
from pathlib import Path
from common.core import read_json

p = argparse.ArgumentParser()
p.add_argument("source")
p.add_argument("target")
a = p.parse_args()
data = read_json(a.source)
if isinstance(data, list):
    if any(row.get("kind") == "audio_chunk_notes" for row in data):
        raise ValueError("音频模型笔记不是逐字稿，不能用于 CER/WER")
    text = "\n".join(row["text"] for row in data)
else:
    text = data["overview"] + "\n" + "\n".join(row["text"] for k in ("decisions", "actions", "risks") for row in data[k])
Path(a.target).parent.mkdir(parents=True, exist_ok=True)
Path(a.target).write_text(text, encoding="utf-8")
