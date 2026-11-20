# Meeting Summary ASR 切换为 SenseVoiceSmall

## 完成内容

- `common/asr.py` 的默认 ASR 已从 faster-whisper 切换为 FunASR SenseVoiceSmall。
- `sensevoice-small` 别名优先读取 `models/weights/SenseVoiceSmall`，并配套
  `models/weights/fsmn-vad`。
- 启用 SenseVoice `output_timestamp=True`，读取 CTC 强制对齐返回的毫秒时间戳。
- 时间戳转换为秒，过滤标点、emoji 和事件符号；词/句关系仍用于路线一说话人归属。
- `run.py`、`scripts/transcribe.py`、路线一、路线三和网页入口均已使用新默认模型。
- 运行记录新增后端、模型 ID、模型路径、VAD 模型、设备和时间戳来源。

## 已验证

### 英文 AMI 片段

- 音频：14.18 秒 AMI IHM 英文片段。
- 模型加载 + 识别：约 10.85 秒，首次运行包含模型冷启动。
- 输出：3 个片段，包含词级时间戳。
- 结果目录：`meeting_summary/outputs/sensevoice_default_20261120/`

命令：

```powershell
python -m scripts.transcribe `
  --audio data/processed/ami_ihm_test_20261120/audio/AMI_EN2002a_H03_MEE071_0146991_0148409.wav `
  --language en `
  --out outputs/sensevoice_default_20261120
```

### 中文路线一完整语音阶段

- 音频：11.39 秒中文会议片段。
- SenseVoice：63 个词级时间戳。
- 说话人切分与聚类：sherpa-onnx，使用 3D-Speaker ERes2Net 中文声纹模型。
- 输出：3 个带匿名说话人/未知标签的证据片段。
- 结果目录：`meeting_summary/outputs/sensevoice_route1_pipeline_20261120/`

该目录包含：

- `asr_words.json`
- `asr_segments.json`
- `diarization.json`
- `speaker_words.json`
- `speech_run.json`
- `segments.json`

这说明 SenseVoice → 词时间戳 → 说话人日志 → 词级归属 → 摘要输入片段已经跑通。

### 网页入口与测试

- 项目隔离虚拟环境 `.venv` 已补齐 `streamlit`、`openai`、`python-dotenv`。
- Streamlit 在 `127.0.0.1:8517` 启动成功，健康检查返回 `ok`。
- 全部项目单元测试：**22 项通过**。

## 摘要阶段

项目当前没有 `.env`，环境中也没有 `TEXT_API_*` 配置，因此未调用远程文本摘要 API。
语音识别和说话人阶段均已本地执行成功；配置合法文本 API 后，
`run.py --route 1` 会继续生成 `summary.json` / `summary.md`。

## 快速运行

```powershell
cd meeting_summary
.\.venv\Scripts\python.exe -m scripts.transcribe --audio meeting.wav --language zh --out outputs/asr_try
.\.venv\Scripts\python.exe run.py --route 1 --audio meeting.wav --language zh --num-speakers 4 --out outputs/r1_try
```

`--asr-model` 可省略，默认为 `sensevoice-small`。
