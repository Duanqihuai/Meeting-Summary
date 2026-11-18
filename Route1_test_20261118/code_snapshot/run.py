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
    model_root = Path(__file__).parent / "models/diarization"
    p.add_argument("--no-diarization", action="store_true", help="路线一关闭说话人日志，复现旧基线")
    p.add_argument("--segmentation-model", default=str(model_root / "model.onnx"))
    p.add_argument("--embedding-model", default=str(model_root / "nemo_en_titanet_small.onnx"))
    p.add_argument("--num-speakers", type=int, default=-1, help="已知人数可指定；-1按阈值自动估计")
    p.add_argument("--cluster-threshold", type=float, default=0.5)
    p.add_argument("--diarization-threads", type=int, default=4)
    p.add_argument("--resume-summary", action="store_true", help="仅路线一：校验音频哈希后复用已保存语音结果，重新生成摘要")
    p.add_argument("--skip-final-review", action="store_true", help="关闭路线一的全文证据复核（长会议需人工复核）")
    args = p.parse_args()
    if args.budget < 1000:
        p.error("--budget 至少为 1000")
    if args.route in ("1", "2") and not args.audio:
        p.error("此路线需要 --audio")
    if args.route == "3" and not args.manifest:
        p.error("路线三需要 --manifest")
    if args.num_speakers == 0 or args.num_speakers < -1 or args.diarization_threads < 1:
        p.error("说话人数应为-1或正整数；线程数应为正整数")
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
    if args.resume_summary and args.route != "1":
        p.error("--resume-summary 仅支持路线一")
    if out.exists() and any(out.iterdir()) and not args.resume_summary:
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
                if args.resume_summary:
                    extra = read_json(out / "speech_run.json")
                    if extra.get("source_sha256") != digest(args.audio):
                        raise ValueError("恢复失败：原音频哈希缺失或与当前输入不一致")
                    segments = read_json(out / "segments.json")
                    validate_segments(segments)
                    meta["speech_stages_reused"] = True
                else:
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
            meta.update(extra)
            write_json(out / "segments.json", segments)
            write_json(out / "run.json", meta)
            meta.update(text_model=text_client.model, text_base_url=text_client.base)
            meta["full_transcript_review"] = args.route == "1" and not args.skip_final_review
            summary = summarize(segments, text_client, args.budget, final_review=meta["full_transcript_review"])
        source = args.audio if args.route in ("1", "2") else args.manifest if args.route == "3" else args.transcript
        meta.update(source=str(source), source_sha256=digest(source), status="completed",
                    elapsed_seconds=time.perf_counter() - started)
        if args.resume_summary:
            meta["summary_resume_elapsed_seconds"] = meta["elapsed_seconds"]
            meta["elapsed_seconds"] += meta.get("asr_seconds_including_load", 0) + meta.get("diarization", {}).get("elapsed_seconds", 0)
            meta["elapsed_definition"] = "成功语音阶段耗时＋本次摘要恢复耗时，不含先前失败与人工调试等待"
        save_result(out, segments, summary, meta, [call for c in clients for call in c.calls])
        print(f"已生成：{out.resolve() / 'summary.md'}")
    except Exception as exc:
        meta.update(status="failed", error_type=type(exc).__name__, elapsed_seconds=time.perf_counter() - started)
        write_json(out / "run.json", meta)
        write_json(out / "raw_calls.json", [call for c in clients for call in c.calls])
        raise


if __name__ == "__main__":
    main()
