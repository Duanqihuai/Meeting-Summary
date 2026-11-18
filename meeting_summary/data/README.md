# 已准备数据与来源

`demo/` 是本项目人工编写的中文演示转写与摘要，没有对应真实录音，不用于效果评测。

`raw/ami/ES2002a/` 是从 AMI 官方服务器下载的 ES2002a 混音和四条头戴麦录音。`source.json` 含下载URL、文件大小和SHA256。`tracks.json` 可直接用于路线三。`smoke_60s.wav` 和 `headset_*_60s.wav` 是各对应原始文件的前60秒截取，许可证和署名沿用源数据。

`raw/ami_public_manual_1.6.2.zip` 是官方手工标注包。`processed/ES2002a/` 是脚本转换结果，包含参考转写 JSON、纯文本和参考摘要；转换依据原始时间戳，不以 ASR 输出代替人工标注。仅对选定会议导入，不代表已整理整个 AMI。

**署名：AMI Meeting Corpus, AMI Consortium。许可：CC BY 4.0（依据当前官方发布页面）。** 官方下载与许可证说明：https://groups.inf.ed.ac.uk/ami/download/ 。引用数据时保留来源和派生处理说明。

ES2002a 当前定位为开发/链路验证会议，完整会议与其任何切片均不得同时出现在最终测试集。论文级划分应另外冻结并记录。
