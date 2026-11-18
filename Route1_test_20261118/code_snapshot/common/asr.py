"""faster-whisper 本地推理；默认 CPU int8，首次运行需要下载权重。"""
def load_model(name="small", device="cpu"):
    from faster_whisper import WhisperModel
    return WhisperModel(name, device=device, compute_type="int8" if device == "cpu" else "float16")


def transcribe(model, path, language="zh", speaker=None, offset=0.0):
    result, info = model.transcribe(str(path), language=None if language == "auto" else language,
                                   beam_size=5, vad_filter=True, condition_on_previous_text=False)
    rows = []
    for s in result:
        # 解码器可能把末尾时间预测到音频结束之后，裁剪到真实范围。
        begin, end = max(0.0, s.start), min(info.duration, s.end)
        if s.text.strip() and end > begin:
            rows.append({"start": round(begin + offset, 3), "end": round(end + offset, 3),
                         "speaker": speaker, "text": s.text.strip()})
    return rows, info.duration


def assign_ids(rows):
    return [dict(row, id=f"s{i:05d}") for i, row in enumerate(sorted(rows, key=lambda x: x["start"]), 1)]


def transcribe_words(model, path, language="zh"):
    """路线一使用词级时间戳，避免整句跨越换人点时被全部归给同一个人。"""
    result, info = model.transcribe(str(path), language=None if language == "auto" else language,
                                   beam_size=5, vad_filter=True, condition_on_previous_text=False,
                                   word_timestamps=True, temperature=0.0)
    words, raw = [], []
    for i, s in enumerate(result):
        raw.append({"start": max(0.0, s.start), "end": min(info.duration, s.end), "text": s.text})
        for w in s.words or []:
            start, end = max(0.0, w.start), min(info.duration, w.end)
            if w.word.strip() and end >= start and start < info.duration:
                words.append({"start": round(float(start), 3), "end": round(float(end), 3), "text": w.word,
                              "asr_probability": round(float(w.probability), 4)})
        if i % 40 == 0:
            print(f"ASR进度：{s.end:.1f}/{info.duration:.1f} 秒", flush=True)
    return words, raw, info.duration
