# -*- coding: UTF-8 -*-
from ultralytics import YOLO
import os

if __name__ == '__main__':
    # 定义所有数据集的 key 与对应的 YAML 配置文件名
    datasets = {
        "color": "color.yaml",
        "shape": "shape.yaml",
        "coating_color": "coating_color.yaml",
        "coating_texture": "coating_texture.yaml",
        "condition": "condition.yaml",
        "status": "status.yaml",
        "vein": "vein.yaml"
    }

    # 遍历每个数据集进行训练
    for dataset_key, yaml_file in datasets.items():
        # 创建对应的保存训练结果的目录
        project_dir = f"runs_{dataset_key}"
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        # 加载预训练模型，比如 yolov8s
        model = YOLO("yolov8s.pt")

        print(f"开始训练 {dataset_key} 数据集，配置文件: {yaml_file}")

        # 调用训练函数，设置超参数（根据需要调整）
        results = model.train(
            data=yaml_file,        # 数据集配置文件
            epochs=30,             # 训练轮数
            batch=10,              # 批量大小
            imgsz=640,             # 图片尺寸
            workers=0,             # 数据加载线程数
            amp=False,             # 是否开启自动混合精度
            project=project_dir,   # 保存结果的文件夹，比如 "runs_color"
            name="train_results",  # 训练结果的子文件夹名称
        )

        print(f"{dataset_key} 数据集训练完成，结果：")
        print(results)
