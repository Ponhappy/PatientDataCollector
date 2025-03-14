# -*- coding: UTF-8 -*-
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 创建保存训练结果的目录（如果不存在）
    project_dir = "../runs/runs_coating"
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    # 加载预训练模型 yolov8s.pt
    model = YOLO("yolov8s")

    # 训练模型，设置保存路径
    results = model.train(
        data="../cfg/coating.yaml",  # 配置文件
        epochs=30,  # 训练轮数
        batch=10,  # 批量大小
        imgsz=640,  # 图片尺寸
        workers=0,  # 数据加载线程数
        amp=False,  # 是否开启自动混合精度
        project=project_dir,  # 保存结果的文件夹，指定为 "runs_coating"
        name="train_results",  # 训练结果的子文件夹名字
    )

    # 输出训练结果
    print(results)
