"""说话人日志：神经网络切分 + 声纹向量聚类 + ASR词时间对齐。

这不是声源分离。模型标签只在本次录音内有效，不代表真实姓名。
"""
from collections import defaultdict
from pathlib import Path
import time
from common.core import digest


def diarize(audio_path, segmentation_model, embedding_model, num_speakers=-1, threshold=0.5, threads=4):
    import sherpa_onnx
    from faster_whisper.audio import decode_audio
    for path in [segmentation_model, embedding_model]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"缺少说话人模型：{path}；先运行 python -m scripts.download_diarization_models")
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_model), window_shift_ratio=0.1), num_threads=threads, provider="cpu"),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding_model), num_threads=threads, provider="cpu"),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    if not config.validate():
        raise ValueError("说话人模型配置无效")
    begin = time.perf_counter()
    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    samples = decode_audio(str(audio_path), sampling_rate=engine.sample_rate)
    duration = len(samples) / engine.sample_rate
    last_report = [-1]
    def progress(done, total):
        bucket = int(done / max(total, 1) * 10)
        if bucket != last_report[0]:
            print(f"说话人模型进度：{min(bucket * 10, 100)}%", flush=True)
            last_report[0] = bucket
        return 0
    result = engine.process(samples, callback=progress).sort_by_start_time()
    labels, turns = {}, []
    for s in result:
        if s.speaker not in labels:
            labels[s.speaker] = f"SPEAKER_{len(labels) + 1:02d}"
        a, b = max(0.0, s.start), min(duration, s.end)
        if b > a:
            turns.append({"start": round(a, 3), "end": round(b, 3), "speaker": labels[s.speaker]})
    if not turns:
        raise ValueError("说话人模型没有检测到有效发言")
    meta = {"backend": "sherpa-onnx", "segmentation_model": str(segmentation_model),
            "embedding_model": str(embedding_model), "segmentation_sha256": digest(segmentation_model),
            "embedding_sha256": digest(embedding_model), "num_speakers_requested": num_speakers,
            "num_speakers_detected": len(labels), "cluster_threshold": threshold,
            "threads": threads, "elapsed_seconds": time.perf_counter() - begin,
            "label_semantics": "匿名录音内声纹簇；非真实姓名；按首次出现排序"}
    return turns, meta


def union_duration(intervals):
    """同一说话人区间先取并集，避免重叠窗口重复计算支持度。"""
    end, total = None, 0.0
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
        elif b > end:
            total += b - end
        end = max(end, b) if end is not None else b
    return total


def attribute_words(words, turns):
    aligned = []
    for w in words:
        a, b = w["start"], w["end"]
        # 极短/零时长词用20毫秒窗口处理，避免除零。
        if b - a < 0.02:
            mid = (a + b) / 2
            a, b = max(0.0, mid - 0.01), mid + 0.01
        intervals = defaultdict(list)
        for turn in turns:
            left, right = max(a, turn["start"]), min(b, turn["end"])
            if right > left:
                intervals[turn["speaker"]].append((left, right))
        ranked = sorted(((label, union_duration(spans) / (b - a)) for label, spans in intervals.items()),
                        key=lambda pair: (-pair[1], pair[0]))
        best = ranked[0][1] if ranked else 0.0
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        # 不强行把无覆盖或重叠严重的词归给一个人。
        uncertain = bool(best < 0.5 or second >= 0.25)
        aligned.append(dict(w, speaker=None if uncertain else ranked[0][0], speaker_uncertain=uncertain,
            speaker_candidates=[label for label, support in ranked if support >= 0.15],
            speaker_time_support=round(float(best), 3)))
    return aligned


def group_words(words, max_seconds=20):
    """遇到换人、归属不确定、长停顿或20秒上限就分段，保留原词序。"""
    rows, group = [], []
    def flush():
        if not group:
            return
        start, end = group[0]["start"], max(w["end"] for w in group)
        text = "".join(w["text"] for w in group).strip()
        if end > start and text:
            rows.append({"start": start, "end": end, "speaker": group[0]["speaker"], "text": text,
                         "speaker_uncertain": group[0]["speaker_uncertain"],
                         "speaker_candidates": sorted({s for w in group for s in w["speaker_candidates"]})})
        group.clear()
    for w in words:
        if group and (w["speaker"] != group[-1]["speaker"] or w["speaker_uncertain"] != group[-1]["speaker_uncertain"]
                      or w["start"] - group[-1]["end"] > 1.2 or w["end"] - group[0]["start"] > max_seconds):
            flush()
        group.append(w)
    flush()
    return rows
