"""从 sherpa-onnx 官方发布页下载本地说话人模型，保留来源和校验信息。"""
import argparse
from pathlib import Path
import tarfile
import requests
from common.core import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with temporary.open("wb") as f:
                for block in response.iter_content(1024 * 1024):
                    f.write(block)
        temporary.replace(path)
    return {"file": str(path.relative_to(ROOT)), "url": url, "sha256": digest(path), "bytes": path.stat().st_size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", choices=["nemo", "3dspeaker"], default="nemo")
    a = p.parse_args()
    root = ROOT / "models/diarization"
    archive = root / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    records = [download(BASE + "speaker-segmentation-models/" + archive.name, archive)]
    # 仅提取需要的普通文件；不执行归档中的代码、不接受链接或路径穿越。
    with tarfile.open(archive, "r:bz2") as tar:
        for member in tar.getmembers():
            leaf = Path(member.name).name
            if member.isfile() and leaf in {"model.onnx", "model.int8.onnx", "LICENSE", "README.md"}:
                stream = tar.extractfile(member)
                (root / ("segmentation_" + leaf if leaf in {"LICENSE", "README.md"} else leaf)).write_bytes(stream.read())
    name = "nemo_en_titanet_small.onnx" if a.embedding == "nemo" else "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    records.append(download(BASE + "speaker-recongition-models/" + name, root / name))
    write_json(root / f"sources_{a.embedding}.json", {"source_page": "https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html", "files": records})
    print("模型已下载：", root)


if __name__ == "__main__":
    main()
