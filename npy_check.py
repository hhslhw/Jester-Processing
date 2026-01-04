import numpy as np
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # 用于3D可视化

# 设置FFmpeg路径
plt.rcParams['animation.ffmpeg_path'] = r'E:\software\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe'


def load_and_inspect_npy(file_path):
    """加载并检查.npy文件的基本信息"""
    data = np.load(file_path)
    print(f"数据形状: {data.shape} (N, C, T, V)")
    print("前两个样本的前两帧关键点数据:")
    print(data[:2, :, :2, :])  # 显示完整数据结构
    return data


def visualize_keypoints_sequence(data, sample_idx=0, output_video_path="keypoint_animation.mp4",
                                 scale_factor=500, enable_3d=True, fps=15, wrist_fixed=True):
    """
    可视化某个样本的关键点序列（支持手腕固定模式）
    :param data: (N, C, T, V) 格式的 numpy 数组
    :param sample_idx: 要可视化的样本索引
    :param output_video_path: 输出视频路径
    :param scale_factor: 坐标缩放因子（世界坐标转像素）
    :param enable_3d: 是否启用3D可视化
    :param fps: 视频帧率
    :param wrist_fixed: 是否启用手腕固定模式
    """
    sample_data = data[sample_idx]  # 形状为 (C, T, V)
    num_frames = sample_data.shape[1]
    joints = sample_data.shape[2]

    # 创建图表
    fig = plt.figure(figsize=(8, 8))

    if enable_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Hand Keypoints (Wrist-Fixed)" if wrist_fixed else "3D Hand Keypoints")
        ax.set_xlim(-0.1 * scale_factor, 0.1 * scale_factor)
        ax.set_ylim(-0.1 * scale_factor, 0.1 * scale_factor)
        ax.set_zlim(-0.1 * scale_factor, 0.1 * scale_factor)
        ax.view_init(elev=15, azim=120)  # 初始视角
    else:
        ax = fig.add_subplot(111)
        ax.set_title("2D Hand Keypoints (Wrist-Fixed)" if wrist_fixed else "2D Hand Keypoints")
        ax.set_xlim(-0.1 * scale_factor, 0.1 * scale_factor)
        ax.set_ylim(-0.1 * scale_factor, 0.1 * scale_factor)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # 与图像坐标系保持一致

    # 初始化散点图
    if enable_3d:
        scat = ax.scatter([], [], [], c=[], cmap='viridis', s=30)
    else:
        scat = ax.scatter([], [], c=[], cmap='viridis', s=50)

    def update(frame):
        # 获取当前帧数据 (C, V)
        frame_joints = sample_data[:, frame, :]

        # 如果启用手腕固定模式，进行坐标转换
        if wrist_fixed:
            wrist_pos = frame_joints[:, 0].reshape(3, 1)  # 获取手腕位置 (3,1)
            frame_coords = frame_joints - wrist_pos  # 转换为相对坐标
        else:
            frame_coords = frame_joints

        # 提取坐标并应用缩放
        x = frame_coords[0, :] * scale_factor
        y = frame_coords[1, :] * scale_factor
        z = frame_coords[2, :] * scale_factor

        if enable_3d:
            # 3D可视化
            scat._offsets3d = (x, y, z)
            scat.set_array(z)  # 使用z值颜色映射
        else:
            # 2D可视化
            scat.set_offsets(np.c_[x, y])
            scat.set_array(z)  # 使用z值颜色映射

        return scat,

    ani = FuncAnimation(fig, update, frames=num_frames, interval=1000 // fps, blit=True)

    # 设置视频写入器（增强兼容性）
    writer = FFMpegWriter(
        fps=fps,
        codec='libx264',
        bitrate=5000,
        extra_args=[
            '-pix_fmt', 'yuv420p',  # 标准像素格式
            '-preset', 'fast',  # 编码速度/压缩率平衡
            '-profile:v', 'baseline',  # 基准配置文件
            '-level', '3.0'  # 兼容性等级
        ]
    )

    print(f"正在写入可视化视频到: {output_video_path}")
    ani.save(output_video_path, writer=writer)
    plt.close(fig)


if __name__ == "__main__":
    # 配置参数
    npy_file_path = "E:/work/gesture_recognition/output_keypoints_v5/1_train.npy"
    output_dir = "npy_visualizations"
    scale_factor = 500  # 坐标缩放因子
    enable_3d = False  # 启用3D可视化
    wrist_fixed = True  # 启用手腕固定模式

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载和检查数据
    keypoint_data = load_and_inspect_npy(npy_file_path)

    # 可视化样本
    output_video_path = os.path.join(output_dir,
                                     f"1-sample_0_{'3d' if enable_3d else '2d'}{'_wristfixed' if wrist_fixed else ''}.mp4")
    visualize_keypoints_sequence(
        keypoint_data,
        sample_idx=0,
        output_video_path=output_video_path,
        scale_factor=scale_factor,
        enable_3d=enable_3d,
        wrist_fixed=wrist_fixed
    )
    print(f"可视化完成，视频已保存至: {output_video_path}")
