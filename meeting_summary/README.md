# 结合大模型的会议语音摘要项目

建议题目：**面向会议记录的语音识别与大模型摘要系统：级联、直接语音理解与独立分轨方法对比**。

本项目包含三条独立路线、共享摘要组件、中文注释、Web 原型、数据准备工具和实验报告框架。真实 API 实验需要你填写自己的密钥；离线演示是人工样例，不能用作实验结果。

本地 ASR 默认使用 **SenseVoiceSmall**（FunASR），并通过 CTC 强制对齐输出词级时间戳。`faster-whisper` 仍保留用于音频解码和旧实验兼容，但不再作为默认识别模型。

路线一已加入自动说话人模型，并完成真实DeepSeek端到端调用。最新结果在 [独立测试说明](../Route1_test_20261118/测试说明.md)，另有 [机器输出与回放](../Route1_test_20261118/查看会议摘要.html) 和明确标注修改依据的 [审阅整理版纪要](../Route1_test_20261118/会议摘要_审阅整理版.md)。

## 先选哪条路线

| 路线        | 文件夹                | 输入与处理                                                   | 适用条件                               | 主要局限                                     |
| ----------- | --------------------- | ------------------------------------------------------------ | -------------------------------------- | -------------------------------------------- |
| 1，推荐主线 | `route1_asr_llm` | 录音 → SenseVoiceSmall＋说话人模型 → DeepSeek/Qwen → 证据摘要 | 需要区分谁说了什么 | 重叠可能漏词，匿名身份可能错分 |
| 2，对照实验 | `route2_audio_llm`  | 录音分块 → Qwen-Omni 笔记 → 文本模型归并                   | 不想部署本地 ASR、希望比较直接听音能力 | 费用、音频上传、遗漏难定位；不是天然解决重叠 |
| 3，增强路线 | `route3_multitrack` | 每位参与者独立录音 → 逐轨 ASR → 时间排序 → 证据摘要       | 能取得独立麦克风/在线会议分轨录音      | 增加采集条件，串音仍可能重复识别             |

路线二的“直接”指第一阶段直接理解音频，没有显式逐字转写模型；长会议仍需要分块和文本归并。路线三使用已知轨道身份，不宣称实现盲源分离或自动声纹识别。普通双声道/八通道麦克风阵列不等于不同人的独立录音。完整选型论证见 [方案与数据集](docs/01_方案与数据集.md)。

## 快速开始（Windows PowerShell）

在本目录打开终端。推荐 Python 3.11；直接调用虚拟环境解释器，不需要修改 PowerShell 执行策略。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.template .env
# 用编辑器填写 .env；不要把真实密钥上传到作业仓库。
```

零密钥演示只依赖 Python 标准库：

```powershell
python run.py --route demo --out outputs/demo_first
```

每次指定一个新的输出目录，防止覆盖实验记录。示例同时演示了“拒绝提案不能误记成最终决策”和“相对日期不擅自转为绝对日期”。

启动网页：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

只检验本地识别（不需要密钥，不生成大模型摘要）：

```powershell
.\.venv\Scripts\python.exe -m scripts.transcribe --audio data/raw/ami/ES2002a/smoke_60s.wav --asr-model sensevoice-small --language en --out outputs/asr_local_try
```

默认 ASR 为本地 SenseVoiceSmall，并通过 FSMN-VAD 分块。`sensevoice-small`
会优先读取 `models/weights/SenseVoiceSmall`，也可用 `--asr-model` 指定其他本地
SenseVoice 目录；本地权重缺失时 FunASR 会尝试从 ModelScope 获取。

浏览器打开终端提示的本机地址。支持离线演示、录音上传、三条路线、证据展示与 Markdown/JSON 下载。默认本地 ASR 为 SenseVoiceSmall/CPU；CUDA 需要安装匹配的 PyTorch。

## 真实运行

### 配置文本 API

`.env` 的 `TEXT_API_KEY`、`TEXT_BASE_URL`、`TEXT_MODEL` 必须成套配置。例如 DeepSeek 的地址为 `https://api.deepseek.com`；本次查阅官方文档的示例模型为 `deepseek-flash`。Qwen 可使用账户内可用的文本模型，如 `qwen-plus`，地址从百炼控制台复制，必须与密钥地域匹配。模型名可能变动，因此代码没有硬编码提供商。

路线二额外需要 `AUDIO_API_KEY`、`AUDIO_BASE_URL`、`AUDIO_MODEL`。示例使用 `qwen3-omni-flash`，用流式 Chat Completions 接口、音频输入、纯文本输出。以账户实际可用模型和当前限制为准。无需 OpenAI 账户；这里只使用兼容协议的 Python SDK。

### 路线一

路线一现已默认启用说话人模型。先运行 `python -m scripts.download_diarization_models` 下载模型（默认英语声纹模型，适合本次AMI）。中文可下载 `--embedding 3dspeaker` 并通过 `--embedding-model models/diarization/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` 指定。已知人数可设置 `--num-speakers 4`；`--no-diarization` 关闭新功能。详见 [增强说明](docs/04_路线一说话人增强.md)。

```powershell
.\.venv\Scripts\python.exe run.py --route 1 --audio data/raw/meeting.wav --language zh --asr-model sensevoice-small --out outputs/r1_001
```

SenseVoice 通过 `output_timestamp=True` 返回 CTC 强制对齐的词级时间戳，路线一据此完成说话人归属。首次运行且本地权重缺失时会联网下载；网络不便时把包含 `model.pt` 的目录传给 `--asr-model`。`--device cuda` 需要匹配的 PyTorch CUDA 运行时，CPU 模式无须 GPU。

### 路线二

输入要求 16kHz、单声道、PCM16 WAV。如已安装 FFmpeg：

```powershell
ffmpeg -i input.mp3 -ac 1 -ar 16000 -c:a pcm_s16le data/raw/meeting.wav
.\.venv\Scripts\python.exe run.py --route 2 --audio data/raw/meeting.wav --chunk-seconds 60 --out outputs/r2_001
```

默认60秒一块，交叠2秒，逐块处理；60秒 PCM16 约1.92MB（Base64 后增大）。按当前模型限制调整块大小。交叠减轻切断词句的风险，但不能消除跨块信息遗漏；归并阶段要求去重。证据编号 `a00001` 指向整个音频块，笔记不是逐字稿，不能拿来计算 CER/WER。

### 路线三

参考 `data/tracks.example.json` 建立清单；其中路径相对清单文件所在目录。每条轨道只代表一个参与者。`offset_seconds` 表示该文件起点相对于会议共同起点的非负秒数；负偏移须先裁剪。设备时钟漂移须先校准。

```powershell
.\.venv\Scripts\python.exe run.py --route 3 --manifest data/tracks.example.json --language zh --out outputs/r3_001
```

建议优先使用耳麦近讲录音、关闭扬声器回放。独立头戴麦仍有串音，代码保留各轨结果以便分析，不进行可能删除真实重复确认的全局文本去重。

### 人工转写上限对照

```powershell
.\.venv\Scripts\python.exe run.py --route text --transcript data/demo/segments.json --out outputs/text_001
```

该命令会真实调用文本 API。替换为自己的参考转写即可估计“没有 ASR 错误时摘要能达到什么效果”。它不计作完整语音实验。片段格式为 `id/start/end/speaker/text`；没有真实时间时 start/end 同时写 null，未知说话人写 null。

## 数据与实验

```powershell
# 一场公开会议混音；增加 --tracks 可下载四条头戴麦录音
.\.venv\Scripts\python.exe -m scripts.download_ami --meeting ES2002a
.\.venv\Scripts\python.exe -m scripts.download_ami --meeting ES2002a --tracks
# 从官网取得手工标注 ZIP 后导入，无须手工解压
.\.venv\Scripts\python.exe -m scripts.prepare_ami --annotations data/raw/ami_public_manual_1.6.2.zip --meeting ES2002a
.\.venv\Scripts\python.exe run.py --route 1 --audio data/raw/ami/ES2002a/ES2002a.Mix-Headset.wav --language en --out outputs/ami_r1
.\.venv\Scripts\python.exe run.py --route 3 --manifest data/raw/ami/ES2002a/tracks.json --language en --out outputs/ami_r3
```

下载工具记录 URL、许可证说明、大小、SHA256；已有文件不重复下载。默认 AMI 样例用于调试，不是完整测试集。详见 [数据选择](docs/01_方案与数据集.md)、[实验设计](docs/02_实验设计与评测.md)、[报告草稿](docs/03_实验报告草稿.md)。

每次成功生成 `segments.json`、`summary.json`、`summary.md`、`run.json`、`raw_calls.json`。引用检查只确保编号存在，不保证事实正确。实际模型回复、token 用量、总耗时保留在本地；失败时写 `status=failed`，不伪造成功摘要。原始响应和转写包含会议内容，提交前按实际数据授权处理。

```powershell
.\.venv\Scripts\python.exe -m scripts.export_text outputs/ami_r1/segments.json outputs/asr.txt
.\.venv\Scripts\python.exe -m scripts.evaluate --task asr --unit word --reference data/processed/ES2002a/reference_transcript.txt --prediction outputs/asr.txt --out outputs/asr_metrics.json
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 提交材料与当前边界

作业 PDF 要求在 **2026年12月17日前**提交数据集、源代码、模型和实验报告；汇报时间为12月3日和12月10日。报告应包含方法、数据、结果分析、原型展示和总结。实验报告60分、汇报30分、考勤10分。PDF 的11月15日前确定分组时间已早于本次整理日期，请自行核对课程安排。

`models/MODEL_CARD.md` 记录模型交付办法；闭源 API 没有可提交的权重，不能拿一份 API 名称说明冒充模型文件。建议提交本地 ASR 权重或可复现获取方式，并向教师确认 API 模型是否满足“模型”提交要求。

此前的通用报告表仍是待测模板；路线一的最新真实端到端实验单独保存在上一级 `Route1_test_20261118`，以该目录的实际运行记录和说明为准，不能将单次结果填成多路线完整对比。自动化测试与初始本机验证记录见 `docs/验证记录.md`。
