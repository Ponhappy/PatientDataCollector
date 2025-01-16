# tongue_detection.py

from ultralytics import YOLO

def detect_tongue(img, model):
    results = model(img)
    if results and len(results.xyxy[0]) > 0:  # 假设至少有一个检测到的舌象
        return True, results
    else:
        return False, None
