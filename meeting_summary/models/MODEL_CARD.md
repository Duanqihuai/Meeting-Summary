# 模型说明与交付

本项目复用预训练模型，未声称自行训练。

路线一已加入本地说话人模型：`models/diarization/model.onnx`（pyannote segmentation 3.0的官方sherpa导出版）和 `nemo_en_titanet_small.onnx`（本次英语声纹模型）。来源及SHA256见 `models/diarization/sources_nemo.json`；切分模型MIT许可在同目录 `segmentation_LICENSE`。匿名标签由聚类生成，不是实名身份识别。详见 `docs/04_路线一说话人增强.md`。

本地 ASR：faster-whisper（CTranslate2 推理），默认多语种 small；更高质量可选 large-v3。CPU int8，GPU float16。实际模型身份、路径、设备记录在 run.json。首次运行会下载，需保存实际权重文件和来源/版本；离线环境用 `--asr-model models/weights/实际目录`。

获取方式示例（安装依赖后）：

```python
from faster_whisper.utils import download_model
# 这是实际权重下载，可能耗时并占用磁盘。
download_model("small", output_dir="models/weights/faster-whisper-small")
```

文本模型：DeepSeek或Qwen的OpenAI兼容服务，配置在 `.env`；音频模型：Qwen-Omni，纯文本流式输出。实际可用模型与地域从账户控制台确认。为减少长会议截断，每次文本输出上限4096 token，单次输入默认12000字符预算；字符预算不是模型token计数。

闭源 API 无法交付权重。课程“提交模型”应交付可分发的本地ASR权重、来源许可证、环境与模型参数说明，并明确说明摘要模型为远程服务；能否只交获取脚本由教师确认。不得提交个人 API Key，也不能宣称 API 名称等价于权重。

为了复现，保存依赖锁、模型文件 SHA256、下载日期、API实际模型名与请求结果；同名 API 后台可能升级，不能保证跨日期逐字一致。

官方实现参考：[faster-whisper](https://github.com/SYSTRAN/faster-whisper)。接口参考：[DeepSeek](https://api-docs.deepseek.com/)、[Qwen-Omni](https://help.aliyun.com/zh/model-studio/qwen-omni)。核查日期2026-11-18。
