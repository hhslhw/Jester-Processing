import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from st_gcn.net.st_gcn_hand_v3 import Model


# ---------------------- 维度验证工具 ----------------------
def debug_dimensions(tensor, stage):
    """打印各阶段维度信息"""
    print(f"[Debug] {stage} 维度: {tensor.shape}")


# ---------------------- 模型加载 ----------------------
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(channel=3, num_class=4, window_size=35).to(device)

    # 严格状态加载
    checkpoint = torch.load("1best_model.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=True)  # 开启strict模式

    # 验证data_bn参数
    assert model.data_bn.num_features == 63, "BN参数维度异常！"
    return model, device


# ---------------------- 数据处理 ----------------------
class DataProcessor:
    def __init__(self, window_size=35):
        self.window_size = window_size
        # 修改缓冲区为 (T, V, C) 格式
        self.buffer = np.zeros((window_size, 21, 3))  # 注意维度顺序变化

    def update_buffer(self, new_frame):
        """滑动窗口更新（每帧更新）"""
        # 滚动缓冲区并插入新数据
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1] = new_frame  # new_frame形状应为(21,3)

    def get_model_input(self):
        """生成符合模型要求的连续张量"""
        # 转换为 (C, T, V) 并确保连续性
        tensor_data = torch.from_numpy(self.buffer.transpose(2, 0, 1))  # (C,T,V)
        return tensor_data.unsqueeze(0).to(torch.float32).contiguous()  # (1,C,T,V)


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 初始化
    model, device = load_model()
    processor = DataProcessor()
    mp_hands = mp.solutions.hands.Hands()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 关键点提取
        results = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark

            # 提取当前帧数据 (21个点, 每个点xyz)
            current_frame = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # (21,3)

            # 更新缓冲区
            processor.update_buffer(current_frame)

            # 当缓冲区填满时
            if processor.frame_count >= processor.window_size:
                input_tensor = processor.get_model_input().to(device)

                # 添加连续性验证
                assert input_tensor.is_contiguous(), "张量不连续！"

                # 推理
                with torch.no_grad():
                    output = model(input_tensor)

            # 显示结果
            pred_class = torch.argmax(output).item()
            cv2.putText(frame, f"Pred: {pred_class}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Preview', frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
