# 路线二：语音大模型直接理解

入口：项目根目录 `python run.py --route 2 --audio 文件.wav --out outputs/新目录`。

额外配置 AUDIO 三个环境变量。音频模型使用带 Base64 WAV 的 input_audio 和流式文本输出；默认每块60秒、交叠2秒。第一阶段生成笔记而非逐字稿，第二阶段用文本模型统一结构。原始笔记保留在 segments.json，kind 标记为 audio_chunk_notes。

本路线的证据是整个块，不是精确词句对齐；不计算转写CER。重叠语音仍可能漏听，直接听音不代表更准确，必须实际对比。
