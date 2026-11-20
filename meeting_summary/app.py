"""课堂演示原型。通过子进程复用真实 CLI，避免界面与实验流程不一致。"""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.resolve()
st.set_page_config(page_title="会议语音摘要", page_icon="🎙️", layout="wide")
st.title("🎙️ 会议语音摘要")
st.caption("SenseVoiceSmall 本地识别 · 生成会议概述、决策、行动项与可回溯证据。")
mode = st.selectbox("处理方式", ["离线示例", "语音识别＋文本大模型", "语音大模型", "独立参与者分轨"])
audio = None
tracks = []
if mode in ["语音识别＋文本大模型", "语音大模型"]:
    audio = st.file_uploader("上传会议录音（语音大模型要求16kHz单声道PCM16 WAV）", type=["wav"])
    if audio:
        st.audio(audio)
if mode == "独立参与者分轨":
    st.info("每个文件必须对应一位参与者，且已同步到共同起点。文件名将作为说话人标签。")
    tracks = st.file_uploader("上传至少两条独立录音", type=["wav"], accept_multiple_files=True)
language = st.selectbox("录音语言", ["zh", "en", "auto"])
speaker_count = 0
if mode == "语音识别＋文本大模型":
    speaker_count = st.number_input("已知参会人数（0表示自动估计）", min_value=0, max_value=30, value=0)
    st.caption("将自动生成匿名说话人编号。当前默认声纹模型为英语模型；中文可从命令行配置3D-Speaker模型。")
if st.button("生成摘要", type="primary"):
    route = {"离线示例": "demo", "语音识别＋文本大模型": "1", "语音大模型": "2", "独立参与者分轨": "3"}[mode]
    if route in ("1", "2") and not audio:
        st.error("请先上传录音")
        st.stop()
    if route == "3" and len(tracks) < 2:
        st.error("请至少上传两条独立录音")
        st.stop()
    folder = ROOT / "outputs" / ("ui_" + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    out = folder / "result"
    cmd = [sys.executable, str(ROOT / "run.py"), "--route", route, "--out", str(out), "--language", language]
    if route == "1":
        cmd += ["--num-speakers", str(speaker_count or -1)]
    if audio:
        path = folder / "meeting.wav"
        path.write_bytes(audio.getvalue())
        cmd += ["--audio", str(path)]
    if route == "3":
        rows = []
        for i, f in enumerate(tracks):
            name = f"track_{i}.wav"
            (folder / name).write_bytes(f.getvalue())
            rows.append({"speaker": f"{i + 1}_{Path(f.name).stem}", "path": name, "offset_seconds": 0})
        path = folder / "tracks.json"
        path.write_text(json.dumps({"recording_type": "independent_speaker_tracks", "tracks": rows}, ensure_ascii=False), encoding="utf-8")
        cmd += ["--manifest", str(path)]
    with st.spinner("正在处理；本地 ASR 为 SenseVoiceSmall，首次运行可能需要下载权重……"):
        done = subprocess.run(cmd, cwd=ROOT, capture_output=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if done.returncode:
        st.error("处理失败，请检查配置、输入格式和网络。")
        st.code(done.stderr)
    else:
        st.session_state["result_dir"] = str(out)
if "result_dir" in st.session_state:
    out = Path(st.session_state["result_dir"])
    report = (out / "summary.md").read_text(encoding="utf-8")
    st.markdown(report)
    st.download_button("下载 Markdown", report, "会议摘要.md")
    st.download_button("下载 JSON", (out / "summary.json").read_bytes(), "会议摘要.json")
    with st.expander("查看运行记录"):
        st.json(json.loads((out / "run.json").read_text(encoding="utf-8")))
