import os
import cv2
import numpy as np
from ultralytics import YOLO
import cancer_predict
# ------------------ 全局配置 ------------------
#舌苔
CATEGORY_MAP = {
    "薄白润苔": 0,
    "薄白滑苔": 1,
    "白腻苔": 2,
    "白滑腻苔": 3,
    "白粘腻苔": 4,
    "白厚腻苔": 5,
    "白厚松苔": 6,
    "白腻苔化燥": 7,
    "黄腻苔化燥": 8,
    "薄黄苔": 9,
    "黄腻苔": 10,
    "黄粘腻苔": 11,
    "黄滑苔": 12,
    "黄燥苔": 13,
    "黄瓣苔": 14,
    "灰白腻苔": 15,
    "灰黄腻苔": 16,
    "垢腻灰苔": 17,
    "黑腻苔": 18,
    "焦黑苔": 19,
    "黑燥苔": 20,
    "剥苔": 21,
    "腻苔中剥": 22,
    "类剥苔": 23,
    "地图舌": 24,
    "镜面苔": 25,
    "无根苔": 26,
    "薄白苔和黄厚腻苔": 27,
    "黄厚粘腻苔与薄白苔": 28,
    "正常舌": 29
}


# 舌头检测相关配置
DETECTION_MODEL = YOLO("yolov8s.pt")  # 用于判断图片中是否存在舌头的模型
DETECTION_CONF = 0.2  # 舌头检测置信度阈值

# 训练好的癌症预测模型路径（训练代码生成的模型，一般保存在 runs_cancer/train_results/weights/ 下）
CANCER_MODEL_PATH = "../runs/runs_coating/train_results/weights/best.pt"
CANCER_CONF_THRESHOLD = 0.21  # 舌苔类型预测置信度阈值

# ------------------ 舌苔类型预测函数 ------------------
def predict_type_coating(image_path, conf_threshold=CANCER_CONF_THRESHOLD):
    """
    利用训练好的舌苔模型对图片进行预测：
      1. 加载训练好的舌苔预测模型
      2. 对图片进行预测，遍历检测框，选取置信度最高的结果
      3. 输出预测的舌苔类型和对应置信度
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
            print(f"预测舌苔类型: {predicted_category}，置信度: {best_conf:.2f}")
            return predicted_category
        else:
            print("未能预测出舌苔类型。")
    except Exception as e:
        print(f"预测过程中出现异常: {e}")
    return "正常舌"

# ------------------ 综合流程 ------------------
def detect_and_predict_coating(image_path):
    """
    综合流程：
      1. 首先检测图片中是否存在舌头区域
      2. 如果检测到，再将图片输入训练好的癌症模型进行预测
    """
    if cancer_predict.detect_tongue(image_path):
        predicted_category= predict_type_coating(image_path)
        return predicted_category
    else:
        print("由于未检测到舌头，跳过舌苔类型预测。")
    return 0

