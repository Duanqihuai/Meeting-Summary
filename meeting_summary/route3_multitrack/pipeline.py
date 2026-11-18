from pathlib import Path
from common.core import read_json, digest
from common.asr import load_model, transcribe, assign_ids


def run(args):
    path = Path(args.manifest).resolve()
    manifest = read_json(path)
    tracks = manifest.get("tracks", [])
    if manifest.get("recording_type") != "independent_speaker_tracks" or len(tracks) < 2:
        raise ValueError("需要至少两条独立参与者录音，不能使用阵列声道冒充参与者轨道")
    speakers = [t["speaker"] for t in tracks]
    if len(set(speakers)) != len(speakers) or not all(speakers):
        raise ValueError("各轨道 speaker 必须非空且唯一")
    model = load_model(args.asr_model, args.device)
    rows, duration, hashes = [], 0, []
    for track in tracks:
        offset = float(track.get("offset_seconds", 0))
        if offset < 0:
            raise ValueError("请先裁剪到共同起点；offset_seconds 必须非负")
        audio = path.parent / track["path"]
        part, length = transcribe(model, audio, args.language, track["speaker"], offset)
        rows.extend(part)
        duration = max(duration, length + offset)
        hashes.append({"path": str(audio), "sha256": digest(audio), "speaker": track["speaker"]})
    # 不按文本全局去重：不同参与者重复确认也可能是重要证据。
    return assign_ids(rows), {"audio_seconds": duration, "asr_model": args.asr_model,
                              "device": args.device, "tracks": hashes,
                              "speaker_source": "录音轨道元数据，非自动说话人识别"}
