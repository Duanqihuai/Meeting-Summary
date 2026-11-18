# 路线三：独立参与者录音增强

入口：项目根目录 `python run.py --route 3 --manifest data/tracks.example.json --out outputs/新目录`。

清单必须声明 independent_speaker_tracks，每位参与者对应一个文件、唯一标签和非负时间偏移。系统逐轨识别，保留重叠的时间区间，再进行时序摘要。输出同时记录每条录音SHA256。

这是已知轨道身份的处理方案，不是自动说话人日志或声源分离。AMI headsets 可用作此类实验，array channels 不可直接当作参与者。独立麦仍有串音，须报告重复识别。和混音路线比较时必须声明输入条件不同。
