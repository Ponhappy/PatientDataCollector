from PIL import Image
from tongue_detect.YoloModel import YOLO_model

#注意这里将所有警告忽略了
import warnings
warnings.simplefilter("ignore", UserWarning)


if __name__=="__main__":
    # 实例化 YOLO 类
    tongue_model = YOLO_model()
    # 打开单张图片，这里无论是cv2读取还是pil读取都可以，都会进入detect_single_image后变为PIL图片形式
    image_path = '12.jpg'
    image = Image.open(image_path)


    # 调用 detect_single_image 方法进行目标检测
    detected_image,bbox,conf,crop_img= tongue_model.detect_single_image(image,crop=True)

    if detected_image is None:
        print("********************************")
        print("没有舌像")
        print("********************************")
    else:
        print("********************************")
        print("舌像存在")
        print("********************************")
        # # 保存检测后的图片
        save_path = 'detected_image.jpg' 
        detected_image.save(save_path)
        print(f"检测后的图片已保存到 {save_path}")
        # # 保存裁剪后的舌像图片
        crop_path = 'crop_image.jpg' 
        crop_img.save(crop_path)
        print(f"检测后的图片已保存到 {crop_path}")
        # # 打印置信度和边界框
        print(f"置信度为 {conf}")
        print(f"边界框为{bbox}")




    