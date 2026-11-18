"""WAV 流式切块，不把完整会议和所有 Base64 块同时装入内存。"""
import base64
import io
import wave


def chunks(path, seconds=60, overlap=2):
    if not 0 <= overlap < seconds <= 120:
        raise ValueError("需要 0 <= overlap < seconds <= 120")
    with wave.open(str(path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("路线二要求单声道 PCM16 WAV，转换方法见 README")
        rate, total = source.getframerate(), source.getnframes()
        if rate != 16000 or total == 0:
            raise ValueError("请提供非空的 16000 Hz PCM16 WAV")
        step, count = int((seconds - overlap) * rate), int(seconds * rate)
        for index, start in enumerate(range(0, total, step), 1):
            source.setpos(start)
            pcm = source.readframes(count)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(pcm)
            yield f"a{index:05d}", start / rate, min(start + count, total) / rate, "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            if start + count >= total:
                break
