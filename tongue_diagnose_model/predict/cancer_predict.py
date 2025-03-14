import os
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------ 全局配置 ------------------
# 舌头病变类别映射（根据数据集实际类别进行调整）
CATEGORY_MAP = {
    "舌糜": 0,
    "舌疮": 1,
    "重舌": 2,
    "舌菌（舌癌）": 3,
    "舌菌（舌乳头状瘤）": 4,
    "舌血管瘤": 5,
    "舌烂": 6,
    "舌衄": 7,
    "正常舌头": 8
}

# 舌头检测相关配置
DETECTION_MODEL = YOLO("yolov8s.pt")  # 用于判断图片中是否存在舌头的模型
DETECTION_CONF = 0.23  # 舌头检测置信度阈值

# 训练好的癌症预测模型路径（训练代码生成的模型，一般保存在 runs_cancer/train_results/weights/ 下）
CANCER_MODEL_PATH = "../runs/runs_cancer/train_results/weights/best.pt"
CANCER_CONF_THRESHOLD = 0.4  # 癌症类型预测置信度阈值


# ------------------ 舌头检测函数 ------------------
def detect_tongue(image_path):
    """
    检测图片中是否存在舌头（或舌头病变区域）
    利用 DETECTION_MODEL 对图片进行检测，若检测到有效目标，则认为存在舌头。
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print("读取图片失败！")
            return False

        results = DETECTION_MODEL(img)
        boxes = []
        # 遍历所有检测结果
        for result in results:
            for box in result.boxes:
                # 输出检测框信息，便于调试
                print(f"检测框：类别ID={box.cls.item()}, 置信度={box.conf.item()}, 坐标={box.xywh}")
                if box.conf.item() > DETECTION_CONF:
                    # 使用归一化坐标（xyxyn）计算中心点和宽高
                    x_center = (box.xyxyn[0][0] + box.xyxyn[0][2]) / 2
                    y_center = (box.xyxyn[0][1] + box.xyxyn[0][3]) / 2
                    width = box.xyxyn[0][2] - box.xyxyn[0][0]
                    height = box.xyxyn[0][3] - box.xyxyn[0][1]
                    # 过滤面积过小的检测框
                    if width * height > 0.05:
                        boxes.append(box)
        if boxes:
            print("检测到舌头！")
            return True
        else:
            print("未检测到舌头！")
            return False
    except Exception as e:
        print(f"检测过程中出现异常: {e}")
        return False


# ------------------ 舌头癌症类型预测函数 ------------------
def predict_type_cancer(image_path, conf_threshold=CANCER_CONF_THRESHOLD):
    """
    利用训练好的癌症模型对图片进行预测：
      1. 加载训练好的癌症预测模型
      2. 对图片进行预测，遍历检测框，选取置信度最高的结果
      3. 输出预测的舌头癌症类型和对应置信度
    """
    try:
        # 加载训练好的癌症预测模型
        cancer_model = YOLO(CANCER_MODEL_PATH)
        results = cancer_model(image_path, conf=conf_threshold)
        best_conf = 0.0
        predicted_category = None

        # 遍历每个检测结果
        for result in results:
            for box in result.boxes:
                conf = box.conf.item()
                if conf > best_conf:
                    best_conf = conf
                    # 根据模型的 names 映射获取类别名称
                    predicted_category = cancer_model.names.get(int(box.cls), "unknown")

        if predicted_category:
            print(f"预测舌头癌症类型: {predicted_category}，置信度: {best_conf:.2f}")
            return predicted_category
        else:
            print("未能预测出舌头癌症类型。")
    except Exception as e:
        print(f"预测过程中出现异常: {e}")
    return "正常舌头"

# ------------------ 综合流程 ------------------
def detect_and_predict_cancer(image_path):
    """
    综合流程：
      1. 首先检测图片中是否存在舌头（病变）区域
      2. 如果检测到，再将图片输入训练好的癌症模型进行预测
    """
    if detect_tongue(image_path):
        predicted_category= predict_type_cancer(image_path)
        return predicted_category
    else:
        print("由于未检测到舌头，跳过癌症类型预测。")
    return 0


