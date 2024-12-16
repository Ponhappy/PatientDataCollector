import cv2
import os
from datetime import datetime

def capture_and_save_images(output_folder, capture_interval=5, max_images=10):
    """
    从USB摄像头采集图像并保存到指定的文件夹。

    :param output_folder: 图像保存的文件夹路径
    :param capture_interval: 每隔多少秒采集一张图像（默认5秒）
    :param max_images: 最多保存的图像数量
    """
    # 创建输出文件夹，如果不存在的话
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头!")
        return

    image_count = 0
    while image_count < max_images:
        # 读取摄像头的一帧
        ret, frame = cap.read()

        if not ret:
            print("无法读取视频帧!")
            break

        # 获取当前时间，生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = os.path.join(output_folder, f"image_{timestamp}.jpg")

        # 保存图像
        cv2.imwrite(image_filename, frame)
        print(f"图像已保存: {image_filename}")
        
        image_count += 1

        # 暂停，间隔时间控制采集频率
        cv2.waitKey(capture_interval * 1000)  # 等待指定的时间

    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 设置输出文件夹
    output_folder = "captured_images"

    # 调用函数进行图像采集和保存
    capture_and_save_images(output_folder)
