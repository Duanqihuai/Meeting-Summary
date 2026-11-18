"""统一命令行入口：三个真实推理流程，以及明确标记的离线演示。"""
import argparse
from datetime import datetime, timezone
import time
from pathlib import Path
from common.core import read_json, write_json, digest, summarize, save_result, validate_segments


def main():
    p = argparse.ArgumentParser(description="结合大模型的会议语音摘要")
    p.add_argument("--route", choices=["demo", "text", "1", "2", "3"], default="demo")
    p.add_argument("--audio")
    p.add_argument("--transcript", default="data/demo/segments.json")
    p.add_argument("--manifest")
    p.add_argument("--out", default="outputs/demo")
    p.add_argument("--asr-model", default="small", help="模型名称或已下载的本地模型目录")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--language", default="zh", help="zh/en/auto")
    p.add_argument("--chunk-seconds", type=int, default=60)
    p.add_argument("--budget", type=int, default=12000)
    args = p.parse_args()
    if args.budget < 1000:
        p.error("--budget 至少为 1000")
    if args.route in ("1", "2") and not args.audio:
        p.error("此路线需要 --audio")
    if args.route == "3" and not args.manifest:
        p.error("路线三需要 --manifest")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass
    started = time.perf_counter()
    meta = {"route": args.route, "created_utc": datetime.now(timezone.utc).isoformat(),
            "budget_chars": args.budget, "status": "running"}
    out = Path(args.out)
    # 防止失败的新实验被误认为上一次成功结果。
    if out.exists() and any(out.iterdir()):
        p.error("输出目录非空，请使用新的 --out 目录")
    out.mkdir(parents=True, exist_ok=True)
    clients = []
    try:
        if args.route == "demo":
            segments = validate_segments(read_json(args.transcript))
            fixture = Path(__file__).parent / "data/demo/segments.json"
            if segments != read_json(fixture):
                raise ValueError("离线演示仅支持附带示例；真实文本请用 --route text")
            summary = read_json(Path(__file__).parent / "data/demo/summary.json")
            meta["notice"] = "人工编写的演示样例，不调用 ASR/大模型，不是实验结果"
        else:
            from common.client import Client
            text_client = Client("TEXT")
            clients.append(text_client)
            if args.route == "text":
                segments = read_json(args.transcript)
                extra = {"input_kind": "人工逐字稿或外部转写；不代表完整音频流程"}
            elif args.route == "1":
                from route1_asr_llm.pipeline import run
                segments, extra = run(args)
            elif args.route == "2":
                from route2_audio_llm.pipeline import run
                audio_client = Client("AUDIO")
                clients.append(audio_client)
                segments, extra = run(args, audio_client)
            else:
                from route3_multitrack.pipeline import run
                segments, extra = run(args)
            write_json(out / "segments.json", segments)
            meta.update(extra)
            meta.update(text_model=text_client.model, text_base_url=text_client.base)
            summary = summarize(segments, text_client, args.budget)
        source = args.audio if args.route in ("1", "2") else args.manifest if args.route == "3" else args.transcript
        meta.update(source=str(source), source_sha256=digest(source), status="completed",
                    elapsed_seconds=time.perf_counter() - started)
        save_result(out, segments, summary, meta, [call for c in clients for call in c.calls])
        print(f"已生成：{out.resolve() / 'summary.md'}")
    except Exception as exc:
        meta.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=time.perf_counter() - started)
        write_json(out / "run.json", meta)
        write_json(out / "raw_calls.json", [call for c in clients for call in c.calls])
        raise


if __name__ == "__main__":
    main()
