# Jester-Processing
基于mediapipe对高通Jester数据集做简单的关键点检测与可视化，将RGB视频流数据转化为手部骨架数据

## 文件结构 (File Structure)

```bash
.
├── 标注/                          # 开源社区找到的Jester标注（Test未公开）
├── npy_visualizations/            # npy_check的结果展示
├── classfy_dataset.py             # 基于标注划分数据集
├── keypoint_extraction.py         # 对视频数据进行关键点检测，保存结果并在原视频上可视化
└── npy_check.py                   # 对上一个脚本保存的关键点npy文件进行2D与3D可视化
