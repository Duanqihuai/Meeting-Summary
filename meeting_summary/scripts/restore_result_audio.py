"""恢复GitHub结果页所需的AMI录音；原始音频不放入Git仓库。"""
import argparse
from pathlib import Path
import shutil
from common.core import digest, read_json
from scripts.download_diarization_models import download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", default="../Route1_test_20261118")
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    result = Path(args.result).resolve()
    expected = read_json(result / "run/run.json")["source_sha256"]
    source = project / "data/raw/ami/ES2002a/ES2002a.Mix-Headset.wav"
    target = result / "meeting.wav"
    if target.exists():
        if digest(target) != expected:
            raise ValueError("已有回放音频与实验SHA256不一致，请保留备份后重新获取")
        print("回放音频已存在且校验通过")
        return
    if not source.exists():
        download("https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav", source)
    if digest(source) != expected:
        raise ValueError("下载音频与实验SHA256不一致，停止复制")
    shutil.copyfile(source, target)
    print("已恢复回放音频，SHA256校验通过：", target)


if __name__ == "__main__":
    main()
