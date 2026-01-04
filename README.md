# Jester-Processing
基于mediapipe对高通Jester数据集做简单的关键点检测与可视化，将RGB视频流数据转化为手部骨架数据

## 文件结构 (File Structure)

```bash
.
├── log/                           # 训练日志
├── st-gcn/                        # 模型定义(修改ST-GCN使其适配手部拓扑结构)
├── npy_visualizations/            # npy_checkd的结果展示
├── 标注/                          # 开源社区找到的Jester标注(Test未公开)
├── classfy_dataset.py             # 基于标注划分数据集
├── process_dataset.py             # 数据预处理
├── keypoint_extraction_v5.py      # 对视频数据进行关键点检测，保存结果并在原视频上可视化
├── npy_check/                     # keypoint_extraction_v5.py的关键点进行2D与3D可视化 
├── try_v5.py                      # 训练代码
├── v_log_v2.py                    # 日志可视化
└── qianduan.py                    # 搭建简单的前端窗口，针对单个数据样本实现模型的调用推理
