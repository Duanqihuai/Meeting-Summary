# GitHub版本与本机结果的区别

本目录保留真实模型输出、中文纪要、审阅意见、接口用量、评测数据、失败与初版记录，以及代码快照。模型输出与评测分数没有因为上传而改写。

不上传会议音频、模型权重、Python虚拟环境、缓存或API密钥。原始录音来自AMI Meeting Corpus，署名AMI Consortium，采用官方页面所述CC BY 4.0；来源：https://groups.inf.ed.ac.uk/ami/download/ 。参考转写、派生时间标注及原始下载来源/哈希保留在相应JSON中。

## 如何打开可回放页面

克隆仓库后，先在meeting_summary目录安装依赖，再执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.restore_result_audio --result ../Route1_test_20261118
```

脚本恢复约40.7MB的 `meeting.wav` 并核对原实验SHA256；已有本地原录音则直接复制，无须API密钥。之后在浏览器打开 `查看会议摘要.html`。GitHub文件预览本身不会执行HTML音频播放器。

## 哪些路径是历史信息

run.json、speech_run.json和模型来源记录中的Windows绝对路径是原实验机器的记录，不是要求克隆到相同磁盘。重新执行时以仓库README中的相对路径命令为准。`ASR_tests_20261118` 是此前独立实验目录，本次未上传；其中评测参考词的副本已保留在本目录evaluation/reference_words.json。

实现过程的API调用总量包含成功调用、失败调用、初版和连通性探测，详见api_usage.json。任何输出都不含API Key。使用自己账户复现时，从meeting_summary/.env.template复制到.env后配置凭据。
