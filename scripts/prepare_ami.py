"""将官方 AMI 手工标注 ZIP 转为参考转写和参考摘要，不用于模拟 ASR。"""
import argparse
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from common.asr import assign_ids
from common.core import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True, help="官网 ami_public_manual_*.zip")
    p.add_argument("--meeting", default="ES2002a")
    p.add_argument("--out", default="data/processed/ES2002a")
    a = p.parse_args()
    rows, summaries = [], []
    with zipfile.ZipFile(a.annotations) as z:
        for name in z.namelist():
            leaf = Path(name).name
            if leaf.startswith(a.meeting + ".") and leaf.endswith(".words.xml"):
                speaker = leaf.split(".")[1]
                root = ET.fromstring(z.read(name))
                words = []
                def flush():
                    if words:
                        rows.append({"start": float(words[0].attrib["starttime"]),
                                     "end": float(words[-1].attrib["endtime"]), "speaker": speaker,
                                     "text": " ".join(w.text.strip() for w in words)})
                        words.clear()
                for w in root.iter():
                    if w.tag.split("}")[-1] != "w" or not (w.text or "").strip():
                        continue
                    if "starttime" not in w.attrib or "endtime" not in w.attrib or w.attrib.get("punc") == "true":
                        continue
                    if float(w.attrib["endtime"]) <= float(w.attrib["starttime"]):
                        continue
                    if words and (float(w.attrib["starttime"]) - float(words[-1].attrib["endtime"]) > 0.8
                                  or float(w.attrib["endtime"]) - float(words[0].attrib["starttime"]) > 15):
                        flush()
                    words.append(w)
                flush()
            if leaf == a.meeting + ".abssumm.xml":
                root = ET.fromstring(z.read(name))
                summaries.extend(" ".join(e.itertext()).strip() for e in root.iter()
                                 if e.tag.split("}")[-1] == "sentence")
    if not rows:
        raise ValueError("没有找到该会议 words.xml，请检查 ZIP 和 meeting 编号")
    out = Path(a.out)
    write_json(out / "reference_segments.json", assign_ids(rows))
    (out / "reference_transcript.txt").write_text("\n".join(r["text"] for r in sorted(rows, key=lambda x: x["start"])), encoding="utf-8")
    (out / "reference_summary.txt").write_text("\n".join(summaries), encoding="utf-8")
    print(f"已导入 {len(rows)} 个参考片段，{len(summaries)} 个摘要句子；缺少摘要时不得计算摘要分数")


if __name__ == "__main__":
    main()
