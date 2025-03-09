from PIL import Image
import numpy as np
from tongue_detect.YoloModel import YOLO_model # 请将 your_module_name 替换为实际的模块名
from tongue_seg.predict import seg_tongue
# 实例化 YOLO 类
yolo = YOLO_model()

# 打开单张图片
image_path = '12.jpg'  # 请将 path_to_your_image.jpg 替换为实际的图片路径
image = Image.open(image_path)

# 调用 detect_single_image 方法进行目标检测
crop_img,detected_image,_,_ = yolo.detect_single_image(image,crop=True)


if detected_image is None:
    print("********************************")
    print("没有舌像")
    print("********************************")
else:
    print("********************************")
    print("舌像存在")
    print("********************************")
    # # 保存检测后的图片
    save_path = 'detected_image.jpg'  # 请将 detected_image.jpg 替换为你想要保存的文件名
    detected_image.save(save_path)
    print(f"检测后的图片已保存到 {save_path}")
    crop_path = 'crop_image.jpg'  # 请将 detected_image.jpg 替换为你想要保存的文件名
    crop_img.save(crop_path)
    print(f"检测后的图片已保存到 {crop_path}")
    # 显示检测后的图片（可选）
    # detected_image.show()




    