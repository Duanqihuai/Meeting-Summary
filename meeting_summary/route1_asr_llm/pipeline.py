from pathlib import Path
import time
from common.asr import (
    assign_ids,
    load_model,
    model_info,
    transcribe,
    transcribe_words,
)
from common.core import write_json, digest


def run(args):
    started = time.perf_counter()
    model = load_model(args.asr_model, args.device)
    asr_meta = model_info(model)
    if getattr(args, "no_diarization", False):
        rows, duration = transcribe(model, args.audio, args.language)
        return assign_ids(rows), {
            "audio_seconds": duration,
            "asr_model": asr_meta["model_name"],
            "asr": asr_meta,
            "device": args.device,
            "diarization_enabled": False,
        }
    words, raw, duration = transcribe_words(model, args.audio, args.language)
    out = Path(args.out)
    write_json(out / "asr_words.json", words)
    write_json(out / "asr_segments.json", raw)
    asr_seconds = time.perf_counter() - started
    # 推理阶段结束后释放ASR对象；两类模型顺序运行以降低内存占用。
    del model
    from common.diarization import diarize, attribute_words, group_words
    turns, diary_meta = diarize(args.audio, args.segmentation_model, args.embedding_model,
                               args.num_speakers, args.cluster_threshold, args.diarization_threads)
    write_json(out / "diarization.json", turns)
    aligned = attribute_words(words, turns)
    write_json(out / "speaker_words.json", aligned)
    rows = assign_ids(group_words(aligned))
    uncertain = sum(w["speaker_uncertain"] for w in aligned)
    meta = {"audio_seconds": duration, "asr_model": asr_meta["model_name"],
            "asr": asr_meta, "device": args.device,
            "source": str(Path(args.audio).resolve()), "source_sha256": digest(args.audio),
            "language": args.language, "asr_seconds_including_load": asr_seconds,
            "diarization_enabled": True, "diarization": diary_meta, "asr_word_count": len(words),
            "uncertain_speaker_words": uncertain, "uncertain_speaker_word_ratio": uncertain / len(words) if words else None}
    write_json(out / "speech_run.json", meta)
    return rows, meta
