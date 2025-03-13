# camera_thread.py

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import cv2
import time
from PyQt5.QtMultimedia import QSound
import os
from queue import Queue
from threading import Thread
import numpy as np
from ultralytics import YOLO
from PIL import Image
from tongue_detect.YoloModel import YOLO_model
import warnings
warnings.simplefilter("ignore", UserWarning)

class CameraThread(QThread):
    frame_received = pyqtSignal(object)  # 传递OpenCV帧
    crop_tongue_saved = pyqtSignal(str)      # 传递保存的快照路径
    face_detected = pyqtSignal(bool)      # 传递人脸检测结果
    tongue_detected = pyqtSignal(bool, np.ndarray)  # 传递舌象检测结果
    guidance_message = pyqtSignal(str)    # 发送引导消息信号
    tongue_diagnosis_ready = pyqtSignal(object)  # 舌头诊断准备就绪信号
    crop_tongue_saved = pyqtSignal(str)   # 舌头裁剪图像保存信号
    
    # 添加工作模式常量
    MODE_PREVIEW = 0  # 仅预览模式，不保存图像
    MODE_CAPTURE = 1  # 拍摄模式，定期保存图像并分析

    def __init__(self, camera_index=0, crop_tongue_interval=5, save_folder='snapshot'):
        super().__init__()
        self.crop_tongue_interval = crop_tongue_interval
        self.save_folder = save_folder
        self.running = True
        self.camera_index = camera_index
        self.cap = None  # 延迟初始化摄像头
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 加载舌头检测模型
        try:
            self.tongue_model = YOLO_model()
            print("舌头检测模型加载成功")
        except Exception as e:
            print(f"舌头检测模型加载失败: {str(e)}")
            self.tongue_model = None
        
        # 图像处理相关设置
        self.frame_queue = Queue(maxsize=30)  # 最多缓存30帧
        self.processing_enabled = False  # 是否启用处理
        self.save_enabled = True  # 是否保存图像
        self.save_crop_tongue_image = True  # 是否保存裁剪的舌头图像
        self.frames_to_skip = 10  # 每处理一帧，跳过多少帧
        self.frame_count = 0
        self.processor_thread = None
        self.last_save_time = 0
        self.tongue_crop_count = 0  # 保存的舌头裁剪图像计数
        self.max_tongue_crops = 10  # 最多保存的舌头裁剪图像数量
        
        # 舌头检测相关
        self.tongue_detection_enabled = True
        self.has_tongue = False
        self.guidance_interval = 3  # 提示间隔，秒
        self.conf_threshold = 0.5  # 舌头检测置信度阈值

        # 添加工作模式控制
        self.working_mode = self.MODE_PREVIEW  # 默认为预览模式
        self.preview_scale = 0.75  # 预览图像缩放比例，可提高性能
        self.preview_interval = 0.1  # 预览刷新间隔，秒
        self.last_preview_time = 0

    def start_camera(self):
        """延迟初始化摄像头"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.camera_index}")
                return False
            # 设置分辨率（可选）
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        return True

    def run(self):
        """主线程：动态切换预览和拍摄模式"""
        if not self.start_camera():
            return
            
        # 启动处理线程
        self.processor_thread = Thread(target=self.process_frames)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        
        last_guidance_time = 0
        last_capture_time = 0
        capture_duration = 2  # 检测到舌头后持续拍摄时间(秒)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()
            
            # 向处理队列添加图像
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())

            # 模式处理逻辑
            if self.working_mode == self.MODE_PREVIEW:
                # 预览模式：降低帧率显示缩放图像
                if current_time - self.last_preview_time >= self.preview_interval:
                    preview_frame = cv2.resize(frame, (0,0), fx=self.preview_scale, fy=self.preview_scale)
                    self.frame_received.emit(preview_frame)
                    self.last_preview_time = current_time

                # 显示引导提示
                if self.tongue_detection_enabled and not self.has_tongue:
                    if current_time - last_guidance_time > self.guidance_interval:
                        self.guidance_message.emit("请伸出舌头进行检测")
                        last_guidance_time = current_time

            elif self.working_mode == self.MODE_CAPTURE:
                # 拍摄模式：发送原始帧
                self.frame_received.emit(frame)
                
                # 定期保存图像（示例：每秒1张）
                if current_time - self.last_save_time >= 1.0:
                    self.save_crop_tongue(frame)
                    self.last_save_time = current_time

            time.sleep(0.01)
        
        self.cap.release()

    def process_frames(self):
        """处理线程：从队列获取图像并进行处理、保存"""
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.frame_count += 1
                
                # 只处理部分帧以减少计算量
                if self.frame_count % self.frames_to_skip == 0:
                    # 如果启用了舌头检测，检测舌头
                    if self.tongue_detection_enabled and self.tongue_model is not None:
                        detected, bbox, confidence, crop_image = self.detect_tongue(frame)
                        self.has_tongue = detected
                        
                        # 发送检测结果信号
                        self.tongue_detected.emit(detected, frame.copy())
                        
                        # 如果检测到舌头
                        if detected:
                            # 切换到拍摄模式
                            self.working_mode = self.MODE_CAPTURE
                            
                            # 发送舌头诊断准备就绪信号
                            if crop_image is not None:
                                self.tongue_diagnosis_ready.emit(crop_image)
                                
                                # 保存裁剪的舌头图像
                                # if self.save_crop_tongue_image and self.tongue_crop_count < self.max_tongue_crops:
                                #     crop_path = self.save_crop_tongue(crop_image)
                                #     self.crop_tongue_saved.emit(crop_path)
                                #     self.tongue_crop_count += 1
                                
                                # 如果启用了保存且达到保存间隔，保存图像
                                current_time = time.time()
                                if self.save_crop_tongue_image and (current_time - self.last_save_time >= self.crop_tongue_interval) and self.tongue_crop_count < self.max_tongue_crops:
                                    crop_tongue_path = self.save_crop_tongue(frame)
                                    self.crop_tongue_saved.emit(crop_tongue_path)
                                    self.last_save_time = current_time
                                    self.tongue_crop_count += 1
                        else:
                            # 如果没有检测到舌头，切换回预览模式
                            self.working_mode = self.MODE_PREVIEW
                    
                
            else:
                # 队列为空时短暂休眠
                time.sleep(0.01)

    def detect_tongue(self, frame):
        """舌头检测函数，使用YOLO模型"""
        try:    
            # 调用 detect_single_image 方法进行目标检测
            detected,bbox,conf,crop_img= self.tongue_model.detect_single_image(frame,crop=True)
            print(f"载入detect_tongue函数，检测结果为{detected}")
            return detected,bbox,conf,crop_img
            
        
        except Exception as e:
            print(f"舌头检测出错: {str(e)}")
            return False,None,0, None

    def save_crop_tongue(self, frame):
        """保存图像快照"""
        # 使用时间戳生成唯一的文件名
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        crop_tongue_path = os.path.join(self.save_folder, f"crop_tongue_{timestamp}.jpg")

        # 确保保存目录存在
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # 保存快照
        cv2.imwrite(crop_tongue_path, frame)
        return crop_tongue_path
        
    def set_save_enabled(self, enabled):
        """设置是否保存图像"""
        self.save_enabled = enabled
        
    def set_tongue_detection_enabled(self, enabled):
        """设置是否启用舌头检测"""
        self.tongue_detection_enabled = enabled
        
    def set_save_crop_tongue_enabled(self, enabled):
        """设置是否保存裁剪的舌头图像"""
        self.save_crop_tongue_image = enabled
        
    def set_frames_to_skip(self, count):
        """设置跳过的帧数"""
        if count > 0:
            self.frames_to_skip = count
            
    def set_crop_tongue_interval(self, interval):
        """设置截图间隔（秒）"""
        if interval > 0:
            self.crop_tongue_interval = interval

    def set_mode(self, mode):
        """设置工作模式：预览或拍摄"""
        self.working_mode = mode
        print(f"摄像头工作模式已切换: {'预览' if mode == self.MODE_PREVIEW else '拍摄'}")

    def stop(self):
        """停止线程"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
        self.wait()