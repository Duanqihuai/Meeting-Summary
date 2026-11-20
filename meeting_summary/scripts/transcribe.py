"""只运行本地识别，无需 API Key；用于检查音频和定位识别阶段问题。"""
import argparse
import time
from pathlib import Path
from common.asr import assign_ids, load_model, model_info, transcribe
from common.core import digest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--asr-model", default="sensevoice-small")
    p.add_argument("--language", default="zh")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error("输出目录非空，请指定新目录")
    start = time.perf_counter()
    model = load_model(a.asr_model, a.device)
    info = model_info(model)
    rows, seconds = transcribe(model, a.audio, a.language)
    elapsed = time.perf_counter() - start
    write_json(out / "segments.json", assign_ids(rows))
    write_json(out / "run.json", {"stage": "local_asr_only",
                "model": info["model_name"], "model_id": info["model_id"],
                "model_path": info["model_path"], "vad_model": info["vad_model"],
                "language": a.language, "device": a.device,
                "audio_seconds": seconds, "elapsed_seconds": elapsed,
                "rtf_including_model_load": elapsed / seconds if seconds else None,
                "source_sha256": digest(a.audio), "segment_count": len(rows)})
    print(
        f"转写完成（{info['model_name']}）：{len(rows)} 段，"
        f"录音 {seconds:.1f} 秒，耗时 {elapsed:.2f} 秒"
    )


if __name__ == "__main__":
    main()
