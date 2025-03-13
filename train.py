# -*- coding: UTF-8 -*-
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载脚本目录下的预训练的模型yolov8s.pt
    model = YOLO("yolov8s")

    # Train the model
    results = model.train(data="cfg/yolo.yaml", epochs=100, batch=16, imgsz=640, workers=0, amp=False)
    print(results)
