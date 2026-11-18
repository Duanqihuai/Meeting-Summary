# 路线一：本地识别、说话人日志与文本摘要

入口：项目根目录 `python run.py --route 1 --audio 文件.wav --out outputs/新目录`。

现在默认启用自动说话人日志。首次使用运行 `python -m scripts.download_diarization_models` 下载切分模型和英语声纹模型；已知4人可加 `--num-speakers 4`，未知人数默认按聚类阈值估计。`--no-diarization` 可关闭该功能以复现原基线。

`pipeline.py` 顺序执行词级ASR、神经网络说话人切分、声纹聚类和词时间对齐。输出匿名 `SPEAKER_01` 等标签，换人时分段；模型没有覆盖或多人竞争归属的词标为未知。`common/core.py` 再调用DeepSeek/Qwen生成中文结构化摘要。不是简单按停顿轮流给A/B标签，也不从参考转写复制真实说话人。

详细原理、命令和已知局限见 [路线一说话人增强](../docs/04_路线一说话人增强.md)。

推荐作为主线：能单独测 ASR 与摘要，定位错误最直接。VAD 用来减少静音段，不解决多个人同时说话的问题。
