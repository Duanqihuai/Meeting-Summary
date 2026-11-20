"""FunASR SenseVoiceSmall local ASR with word-level timestamps.

SenseVoice returns CTC-forced-aligned timestamps when ``output_timestamp``
is enabled. Word timestamps are required by route 1 for speaker attribution.
"""

from __future__ import annotations

import os
import re
import unicodedata
import wave
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENSEVOICE_MODEL_ID = "iic/SenseVoiceSmall"
SENSEVOICE_ALIASES = {
    "sensevoice",
    "sensevoice-small",
    "sensevoicesmall",
    SENSEVOICE_MODEL_ID.lower(),
}
SENTENCE_ENDINGS = {".", "!", "?", "。", "！", "？"}
_CJK_RE = re.compile(r"[\u3400-\u9fff]")


class SenseVoiceRunner:
    """Small adapter that keeps the existing ASR call interface."""

    backend = "funasr"
    model_id = SENSEVOICE_MODEL_ID

    def __init__(
        self,
        model: Any,
        name: str,
        model_location: str,
        vad_location: str,
        device: str,
    ) -> None:
        from funasr.utils.postprocess_utils import (
            rich_transcription_postprocess,
        )

        self._model = model
        self.name = name
        self.model_location = model_location
        self.vad_location = vad_location
        self.device = device
        self._postprocess = rich_transcription_postprocess


def _candidate_model_dirs() -> list[Path]:
    candidates = []
    configured = os.environ.get("SENSEVOICE_MODEL_DIR")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            PROJECT_ROOT / "models" / "weights" / "SenseVoiceSmall",
            PROJECT_ROOT / "models" / "weights" / "sensevoice-small",
        ]
    )
    return candidates


def _resolve_model_location(name: str) -> str:
    requested = str(name or "sensevoice-small").strip()
    if requested.lower() in SENSEVOICE_ALIASES:
        for candidate in _candidate_model_dirs():
            if (candidate / "model.pt").is_file():
                return str(candidate.resolve())
        return SENSEVOICE_MODEL_ID

    path = Path(requested).expanduser()
    if path.is_dir():
        if not (path / "model.pt").is_file():
            raise FileNotFoundError(
                f"SenseVoice 目录缺少 model.pt：{path.resolve()}"
            )
        return str(path.resolve())
    raise ValueError(
        "ASR 已切换到 SenseVoiceSmall。请使用 sensevoice-small "
        "或包含 model.pt 的本地模型目录。"
    )


def _resolve_vad_location(model_location: str) -> str:
    candidates = []
    configured = os.environ.get("FSMN_VAD_MODEL_DIR")
    if configured:
        candidates.append(Path(configured))
    local = Path(model_location)
    if local.is_dir():
        candidates.append(local.parent / "fsmn-vad")
    candidates.append(
        PROJECT_ROOT / "models" / "weights" / "fsmn-vad"
    )
    for candidate in candidates:
        if (candidate / "model.pt").is_file():
            return str(candidate.resolve())
    return "fsmn-vad"


def load_model(name="sensevoice-small", device="cpu"):
    """Load SenseVoiceSmall and its optional FSMN-VAD model."""
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "缺少 FunASR。请安装 requirements.txt 中的 funasr/modelscope/"
            "torch/torchaudio 依赖。"
        ) from exc

    location = _resolve_model_location(str(name))
    vad_location = _resolve_vad_location(location)
    resolved_device = "cuda:0" if device == "cuda" else device
    vad_kwargs = {"max_single_segment_time": 30000}
    if vad_location != "fsmn-vad":
        vad_kwargs["model_path"] = vad_location
    model = AutoModel(
        model=location,
        trust_remote_code=False,
        vad_model=vad_location,
        vad_kwargs=vad_kwargs,
        device=resolved_device,
        disable_update=True,
    )
    return SenseVoiceRunner(
        model=model,
        name="sensevoice-small",
        model_location=location,
        vad_location=vad_location,
        device=resolved_device,
    )


def model_info(model: SenseVoiceRunner) -> dict[str, Any]:
    return {
        "backend": model.backend,
        "model_name": model.name,
        "model_id": model.model_id,
        "model_path": model.model_location,
        "vad_model": model.vad_location,
        "device": model.device,
        "word_timestamps": "ctc_forced_alignment",
    }


def _is_cjk(char: str) -> bool:
    return bool(char and _CJK_RE.match(char))


def _is_punctuation(token: str) -> bool:
    return bool(token) and all(
        unicodedata.category(char).startswith("P") for char in token
    )


def _is_lexical(token: str) -> bool:
    return any(
        char.isalnum() or unicodedata.category(char).startswith("L")
        for char in token
    )


def _clean_token(token: Any) -> str:
    value = unicodedata.normalize("NFKC", str(token or ""))
    return value.replace("▁", "").strip()


def _duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        try:
            from faster_whisper.audio import decode_audio

            samples = decode_audio(str(path), sampling_rate=16000)
            return len(samples) / 16000
        except Exception:
            with wave.open(str(path), "rb") as stream:
                return stream.getnframes() / stream.getframerate()


def _timestamp_pairs(result: dict[str, Any]) -> list[list[float]]:
    timestamps = result.get("timestamp") or result.get("timestamps") or []
    pairs = []
    for item in timestamps:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            start, end = item[-2], item[-1]
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            continue
        pairs.append([float(start), float(end)])
    return pairs


def _result_tokens(
    results: list[dict[str, Any]],
    duration: float,
) -> list[dict[str, Any]]:
    """Normalize SenseVoice words and millisecond timestamps to seconds."""
    items: list[dict[str, Any]] = []
    previous_lexical = False
    for result in results:
        words = result.get("words") or []
        timestamps = _timestamp_pairs(result)
        if words and len(words) != len(timestamps):
            raise RuntimeError(
                "SenseVoice 返回的 words 与 timestamp 数量不一致："
                f"{len(words)} != {len(timestamps)}"
            )
        for raw_token, pair in zip(words, timestamps):
            token = _clean_token(raw_token)
            if not token:
                continue
            start = max(0.0, min(float(pair[0]) / 1000.0, duration))
            end = max(start, min(float(pair[1]) / 1000.0, duration))
            punctuation = _is_punctuation(token)
            lexical = _is_lexical(token)
            if not punctuation and not lexical:
                # Ignore emoji, sentiment marks, and other event symbols.
                continue
            text = token
            if lexical and previous_lexical:
                previous = items[-1]["text"][-1:] if items else ""
                if (
                    not _is_cjk(previous)
                    and not _is_cjk(token[0])
                    and previous not in {"'", "’", "-"}
                ):
                    text = " " + text
            items.append(
                {
                    "text": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "lexical": lexical,
                    "sentence_break": punctuation
                    and token[-1] in SENTENCE_ENDINGS,
                }
            )
            if lexical:
                previous_lexical = True
            elif token[-1] in SENTENCE_ENDINGS:
                previous_lexical = False
    return items


def _group_timestamped_items(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    tokens: list[str] = []
    group_start = None
    group_end = None

    def flush() -> None:
        nonlocal tokens, group_start, group_end
        text = "".join(tokens).strip()
        if text and group_start is not None and group_end is not None:
            segments.append(
                {
                    "start": round(group_start, 3),
                    "end": round(group_end, 3),
                    "text": text,
                }
            )
        tokens = []
        group_start = None
        group_end = None

    for item in items:
        if item["lexical"]:
            if (
                group_start is not None
                and (
                    item["start"] - group_end > 0.9
                    or item["end"] - group_start > 15
                )
            ):
                flush()
            if group_start is None:
                group_start = item["start"]
            group_end = max(group_end or item["end"], item["end"])
            tokens.append(item["text"])
            words.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                    "asr_probability": None,
                }
            )
        elif tokens:
            tokens.append(item["text"])
        if item["sentence_break"]:
            flush()
    flush()
    return words, segments


def _decode(
    model: SenseVoiceRunner,
    path: str | Path,
    language: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    audio_path = Path(path).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    duration = _duration_seconds(audio_path)
    results = model._model.generate(
        input=str(audio_path),
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=20,
        merge_vad=True,
        merge_length_s=15,
        output_timestamp=True,
        disable_pbar=True,
    )
    items = _result_tokens(results or [], duration)
    if not items:
        return [], [], duration
    words, segments = _group_timestamped_items(items)
    if words and not segments:
        segments = [
            {
                "start": words[0]["start"],
                "end": max(word["end"] for word in words),
                "text": "".join(word["text"] for word in words).strip(),
            }
        ]
    return words, segments, duration


def transcribe(model, path, language="zh", speaker=None, offset=0.0):
    """Return sentence-like segments for no-diarization or track routes."""
    words, raw, duration = _decode(model, path, language)
    if not words:
        return [], duration
    rows = []
    for segment in raw:
        begin = max(0.0, float(segment["start"]))
        end = min(duration, float(segment["end"]))
        if end > begin and segment["text"].strip():
            rows.append(
                {
                    "start": round(begin + offset, 3),
                    "end": round(end + offset, 3),
                    "speaker": speaker,
                    "text": segment["text"].strip(),
                }
            )
    return rows, duration


def assign_ids(rows):
    return [
        dict(row, id=f"s{i:05d}")
        for i, row in enumerate(
            sorted(rows, key=lambda item: item["start"]), 1
        )
    ]


def transcribe_words(model, path, language="zh"):
    """Run SenseVoice once and return words plus sentence-like raw segments."""
    words, raw, duration = _decode(model, path, language)
    print(
        f"ASR进度：{duration:.1f}/{duration:.1f} 秒，"
        f"{len(words)} 个词级时间戳",
        flush=True,
    )
    return words, raw, duration
