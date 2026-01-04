import sys
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import time
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QProgressBar
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 添加项目根目录到 Python Path
sys.path.append(os.path.abspath("E:/work/ST-GCN"))

# ---------------------- 导入模型类 ----------------------
try:
    from st_gcn.net.st_gcn_hand_v5 import Model
except ImportError as e:
    print(f" 模型导入失败: {e}")
    Model = None


# ---------------------- 关键点提取函数 ----------------------
def extract_keypoints_from_folder(folder_path, output_shape=(3, 35, 21)):
    """将图片文件夹转换为关键点数组"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])[:output_shape[1]]  # 取前35张图片

    frames = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ 无法读取图像: {image_path}")
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_world_landmarks:
            best_idx = max(
                range(len(results.multi_handedness)),
                key=lambda i: results.multi_handedness[i].classification[0].score
            )

            keypoints = []
            for landmark in results.multi_hand_world_landmarks[best_idx].landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])

            frames.append(np.array(keypoints).T)  # (3, 21)

    hands.close()

    if not frames:
        return None

    # 转换为 (C, T, V)
    keypoint_array = np.stack(frames, axis=1)
    return keypoint_array


# ---------------------- 文件夹处理线程 ----------------------
class FolderProcessor(QThread):
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(np.ndarray, object)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            keypoint_data = extract_keypoints_from_folder(self.folder_path)
            if keypoint_data is None:
                raise ValueError("未检测到任何有效手部关键点")
            self.result_signal.emit(keypoint_data, None)
        except Exception as e:
            self.result_signal.emit(None, str(e))


# ---------------------- 模型加载类 ----------------------
class ModelLoader:
    def __init__(self):
        self.model_path = "best_model_9.pth"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.class_labels = {
            0: "动作0", 1: "动作1", 2: "动作2",
            3: "动作3", 4: "动作4", 5: "动作5",
            6: "动作6", 7: "动作7", 8: "动作8"
        }

        self.model = None
        if Model is None:
            print("❌ 模型类未定义")
            return

        try:
            self.model = Model(
                channel=3,
                num_class=9,
                window_size=35,
                num_point=21,
                multiscale=True,
                use_attention=True
            ).to(self.device)

            if os.path.exists(self.model_path):
                self.model.load_state_dict(
                    torch.load(self.model_path, map_location=self.device, weights_only=True)
                )
                self.model.eval()
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def predict(self, keypoint_array):
        """执行模型推理"""
        if self.model is None:
            return -1, 0.0

        input_tensor = torch.from_numpy(keypoint_array).float().unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
            return predicted.item(), confidence.item()
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return -1, 0.0


# ---------------------- 热力图显示组件 ----------------------
class HeatmapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.colorbar = None  # ✅ 新增：用于保存colorbar引用

    def plot_heatmap(self, keypoint_data):
        """绘制关键点热力图（显示X轴数据）"""
        self.ax.clear()
        if keypoint_data is not None:
            im = self.ax.imshow(keypoint_data, cmap='viridis')
            self.ax.set_xlabel('Joint Index')
            self.ax.set_ylabel('Time Frame')

            # ✅ 如果已有 colorbar，则更新；否则新建
            if self.colorbar:
                self.colorbar.update_normal(im)
            else:
                self.colorbar = self.figure.colorbar(im, ax=self.ax)
        self.draw()


# ---------------------- 主界面类 ----------------------
class GestureRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_loader = ModelLoader()
        self.keypoint_data = None
        self.start_time = None  # ✅ 记录推理开始时间
        self.init_ui()

    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout()

        # 文件夹选择区域
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("未选择文件夹")
        select_btn = QPushButton("选择文件夹")
        select_btn.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(select_btn)

        # 控制按钮区域
        control_layout = QHBoxLayout()
        self.process_btn = QPushButton("处理文件夹")
        self.process_btn.clicked.connect(self.process_folder)
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        control_layout.addWidget(self.process_btn)
        control_layout.addWidget(self.save_btn)

        # 热力图显示
        self.heatmap_canvas = HeatmapCanvas()

        # 分类结果显示
        self.result_label = QLabel("分类结果: -")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        # 进度条
        self.progress_bar = QProgressBar()

        # 组合布局
        main_layout.addLayout(folder_layout)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(self.heatmap_canvas)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(control_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('手势识别系统')
        self.resize(800, 600)

    def select_folder(self):
        """选择包含35张图片的文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder_path:
            self.folder_label.setText(folder_path)
            self.folder_path = folder_path
            self.keypoint_data = None
            self.result_label.setText("分类结果: -")
            self.heatmap_canvas.plot_heatmap(np.zeros((35, 21)))  # 初始化热力图

    def process_folder(self):
        """启动文件夹处理线程"""
        if not hasattr(self, 'folder_path'):
            QMessageBox.warning(self, "警告", "请先选择包含35张图片的文件夹！")
            return

        if not os.path.exists(self.folder_path):
            QMessageBox.critical(self, "错误", "文件夹不存在，请重新选择！")
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)

        self.start_time = time.time()  # ✅ 在线程启动前记录开始时间

        self.processor = FolderProcessor(self.folder_path)
        self.processor.progress_signal.connect(self.update_progress)
        self.processor.result_signal.connect(self.handle_processing_result)
        self.processor.start()

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def handle_processing_result(self, keypoint_data, error):
        """处理文件夹处理结果"""
        if error:
            QMessageBox.critical(self, "错误", f"处理失败: {error}")
            self.process_btn.setEnabled(True)
            return

        self.keypoint_data = keypoint_data
        self.progress_bar.setValue(100)

        # ✅ 推理结束时间，在此处计算耗时
        end_time = time.time()
        elapsed_time = end_time - self.start_time

        # 显示热力图（使用X轴数据）
        self.heatmap_canvas.plot_heatmap(self.keypoint_data[0])  # (C, T, V)

        # 执行分类
        if self.model_loader.model:
            pred_class, confidence = self.model_loader.predict(self.keypoint_data)
            if pred_class >= 0:
                class_name = self.model_loader.class_labels[pred_class]
                self.result_label.setText(
                    f"分类结果: {class_name} (置信度: {confidence * 100:.1f}%) | 推理耗时: {elapsed_time:.3f}s"
                )
            else:
                self.result_label.setText("分类失败：模型返回无效结果")
        else:
            self.result_label.setText("模型加载失败，无法分类")

        self.process_btn.setEnabled(True)

    def save_results(self):
        """保存关键点数据"""
        if self.keypoint_data is None:
            QMessageBox.warning(self, "警告", "没有可保存的关键点数据！")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存关键点数据", "", "NumPy 文件 (*.npy)"
        )

        if save_path:
            try:
                np.save(save_path, self.keypoint_data)
                QMessageBox.information(self, "成功", "关键点数据已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")


# ---------------------- 应用入口 ----------------------
def main():
    app = QApplication(sys.argv)
    window = GestureRecognitionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
