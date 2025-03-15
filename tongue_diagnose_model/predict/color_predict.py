import os
import cv2
import numpy as np
from ultralytics import YOLO
from . import cancer_predict  

# ------------------ 全局配置 ------------------
##舌头颜色
CATEGORY_MAP = {
    "正常舌": 0,
    "淡白舌": 1,
    "枯白舌": 2,
    "红舌": 3,
    "舌尖红": 4,
    "红绛舌": 5,
    "青紫舌": 6,
    "淡紫舌": 7,
    "淡紫淤堵舌": 8,
    "紫红舌": 9,
    "绛紫舌": 10,
    "瘀斑舌": 11
}

# 舌头检测相关配置
DETECTION_MODEL = YOLO("yolov8s.pt")  # 用于判断图片中是否存在舌头的模型
DETECTION_CONF = 0.25  # 舌头检测置信度阈值

# 训练好的舌头颜色预测模型路径（训练代码生成的模型，一般保存在 runs_cancer/train_results/weights/ 下）
CANCER_MODEL_PATH = "tongue_diagnose_model/runs/runs_color/train_results/weights/best.pt"
CANCER_CONF_THRESHOLD = 0.15  # 舌颜色类型预测置信度阈值

# ------------------ 舌头颜色类型预测函数 ------------------
def predict_type_color(image_path, conf_threshold=CANCER_CONF_THRESHOLD):
    """
    利用训练好的舌苔模型对图片进行预测：
      1. 加载训练好的颜色预测模型
      2. 对图片进行预测，遍历检测框，选取置信度最高的结果
      3. 输出预测的颜色类型和对应置信度
    """
    try:
        # 加载训练好的颜色预测模型
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
            print(f"预测舌头颜色类型: {predicted_category}，置信度: {best_conf:.2f}")
            return predicted_category
        else:
            print("未能预测出舌头颜色类型。")
    except Exception as e:
        print(f"预测过程中出现异常: {e}")
    return "正常舌"

# ------------------ 综合流程 ------------------
def detect_and_predict_color(image_path):
    """
    综合流程：
      1. 首先检测图片中是否存在舌头区域
      2. 如果检测到，再将图片输入训练好的舌头颜色模型进行预测
    """
    if cancer_predict.detect_tongue(image_path):
        predicted_category= predict_type_color(image_path)
        return predicted_category
    else:
        print("由于未检测到舌头，跳过舌苔类型预测。")
    return 0
   
