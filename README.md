# Meeting Summary · 结合大模型的会议语音摘要

会议录音 → 本地语音识别 → 自动区分说话人 → DeepSeek/Qwen → 带录音证据的中文会议纪要。

当前默认本地 ASR 已切换为 **SenseVoiceSmall**（FunASR），并使用 CTC 强制对齐生成路线一所需的词级时间戳。

## 项目目录

| 目录 | 内容 |
|---|---|
| [meeting_summary](meeting_summary/README.md) | 三条路线、中文注释、网页原型、数据准备、评测工具与文档 |
| [Route1_test_20261118](Route1_test_20261118/测试说明.md) | AMI ES2002a路线一的真实完整流程结果与审阅记录 |

路线一已加入本地说话人切分、声纹聚类和词级对齐。路线二为语音大模型直接理解，路线三为独立参与者分轨识别；后两条尚未完成真实摘要对比实验。

## 实际效果

- [会议纪要：审阅整理版](Route1_test_20261118/会议摘要_审阅整理版.md)
- [DeepSeek机器原始摘要](Route1_test_20261118/最终会议摘要.md)
- [测试说明、耗时和接口用量](Route1_test_20261118/测试说明.md)
- [错误案例与局限](Route1_test_20261118/效果审阅与局限.md)
- [说话人功能的中文解释](meeting_summary/docs/04_路线一说话人增强.md)
- [SenseVoiceSmall 集成验证](SENSEVOICE_INTEGRATION_20261120.md)

本次处理约21分钟英语会议，得到4个匿名说话人簇和中文纪要。记录保留数值识别、时间翻译等真实错误；审阅整理版与机器原始输出明确区分。这是一场开发会议上的验证，不能视为整个数据集或中文会议的效果结论。

## 安装与运行

推荐Python 3.11。在仓库根目录的PowerShell中执行：

```powershell
cd meeting_summary
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.template .env
# 在本机编辑.env，配置自己的API Key、模型和接口地址。
.\.venv\Scripts\python.exe -m scripts.download_diarization_models
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

真实路线一参数、中文声纹模型选择和模型下载见 [使用说明](meeting_summary/README.md)。测试命令：`python -m unittest discover -s tests -v`。

## 还原录音证据回放

仓库不包含原始音频、模型权重、虚拟环境或密钥。安装依赖后，在meeting_summary目录执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.restore_result_audio --result ../Route1_test_20261118
```

然后在本机浏览器打开 `Route1_test_20261118/查看会议摘要.html`，点击证据即可播放对应录音。GitHub预览页不执行播放器。详见 [GitHub版本说明](Route1_test_20261118/GitHub版本说明.md)。

AMI数据署名：AMI Meeting Corpus, AMI Consortium。官方公开信号、转写及部分标注采用CC BY 4.0；来源：[AMI官方下载页面](https://groups.inf.ed.ac.uk/ami/download/)。模型来源与许可记录保留在meeting_summary/models，按各模型许可使用。测试日志中的本机绝对路径仅为实验记录。
