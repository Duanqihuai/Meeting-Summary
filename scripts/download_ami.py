"""下载一场 AMI 的混音/独立头戴麦录音，保留来源与 SHA256。"""
import argparse
import re
import urllib.request
from pathlib import Path
from common.core import digest, write_json


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as out:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                out.write(block)
        part.replace(target)
    return {"file": str(target), "url": url, "sha256": digest(target), "bytes": target.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meeting", default="ES2002a")
    p.add_argument("--tracks", action="store_true", help="额外下载四条头戴麦录音，约160MB")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Z]{2}\d{4}[a-d]?", args.meeting):
        p.error("会议编号格式不正确")
    folder = Path("data/raw/ami") / args.meeting
    base = f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{args.meeting}/audio/"
    kinds = ["Mix-Headset"] + ([f"Headset-{i}" for i in range(4)] if args.tracks else [])
    records = []
    for kind in kinds:
        name = f"{args.meeting}.{kind}.wav"
        print("下载/校验", name, flush=True)
        records.append(download(base + name, folder / name))
    write_json(folder / "source.json", {"dataset": "AMI", "license": "CC BY 4.0",
                "attribution": "AMI Meeting Corpus, AMI Consortium",
                "source_page": "https://groups.inf.ed.ac.uk/ami/download/", "files": records})
    if args.tracks:
        write_json(folder / "tracks.json", {"recording_type": "independent_speaker_tracks", "tracks": [
            {"speaker": f"headset_{i}", "path": f"{args.meeting}.Headset-{i}.wav", "offset_seconds": 0}
            for i in range(4)]})


if __name__ == "__main__":
    main()
