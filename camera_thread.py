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
from datetime import datetime
from PIL import Image

class CameraThread(QThread):
    frame_received = pyqtSignal(object)  # 传递OpenCV帧
    face_detected = pyqtSignal(bool)      # 传递人脸检测结果
    # tongue_detected = pyqtSignal(bool)  # 传递舌象检测结果
    guidance_message = pyqtSignal(str)    # 发送引导消息信号
    # tongue_diagnosis_ready = pyqtSignal(object)  # 舌头诊断准备就绪信号
    crop_tongue_saved_path = pyqtSignal(str)   # 舌头裁剪图像保存信号
    original_frame_saved_path = pyqtSignal(str, str)  # 原始帧保存信号(传递原始帧路径和对应的裁剪图路径)
    max_images_reached = pyqtSignal()  # 当达到最大图像数时发出
    
    # 添加工作模式常量
    MODE_PREVIEW = 0  # 仅预览模式，不保存图像
    MODE_CAPTURE = 1  # 拍摄模式，定期保存图像并分析

    def __init__(self, save_dir,camera_index=0, crop_tongue_interval=5):
        super().__init__()
        self.crop_tongue_interval = crop_tongue_interval
        self.save_dir = save_dir
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
        # self.processing_enabled = False  # 是否启用处理
        # self.save_enabled = True  # 是否保存图像
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

        # 添加诊断完成状态标志
        self.diagnosis_completed = False

        # 添加标志以跟踪是否已发送第一帧图像
        self.first_image_sent = False

        # 面诊相关
        self.face_detection_enabled = False
        self.face_diagnosed = False
        self.face_max_images = 10
        self.face_image_count = 0
        self.face_crop_interval = 15  # 每15帧保存一次

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
                        self.guidance_message.emit("👅请伸出舌头")
                        last_guidance_time = current_time
            

            elif self.working_mode == self.MODE_CAPTURE:
                # 拍摄模式：发送原始帧
                self.frame_received.emit(frame)
                

            # 在处理舌诊之后添加面诊处理代码
            if self.face_detection_enabled and not self.face_diagnosed:
                # 保存第一帧以进行面诊
                if self.face_image_count == 0:
                    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                    face_image_path = os.path.join(self.save_dir, "face_images", f"face_{timestamp}.jpg")
                    cv2.imwrite(face_image_path, frame)
                    self.original_frame_saved_path.emit(face_image_path, "")  # 发送面诊图像路径
                    self.face_image_count += 1
                
                # 继续保存其他9张图像作为备用
                elif self.frame_count % self.face_crop_interval == 0 and self.face_image_count < self.face_max_images:
                    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                    face_image_path = os.path.join(self.save_dir, "face_images", f"face_{timestamp}.jpg")
                    cv2.imwrite(face_image_path, frame)
                    self.face_image_count += 1
                    
                    # 当达到最大图像数量时发出信号
                    if self.face_image_count >= self.face_max_images:
                        self.max_images_reached.emit()

            time.sleep(0.01)
        
        self.cap.release()

    def process_frames(self):
        """处理线程：从队列获取图像并进行处理、保存"""
        print("进入process_frames函数")
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                self.frame_count += 1
                
                # 只处理部分帧以减少计算量
                if self.frame_count % self.frames_to_skip == 0:
                    # 如果诊断未完成且启用了舌头检测，则检测舌头
                    if not self.diagnosis_completed and self.tongue_detection_enabled and self.tongue_model is not None:
                        detected, bbox, confidence, crop_image = self.detect_tongue(frame)
                        self.has_tongue = detected
                        
                        # 如果检测到舌头且没有发送过第一帧图像
                        if detected and not self.first_image_sent:
                            # 切换到拍摄模式
                            self.working_mode = self.MODE_CAPTURE
                                
                            # 如果启用了保存且达到保存间隔，保存图像
                            current_time = time.time()
                            if self.save_crop_tongue_image and self.tongue_crop_count < self.max_tongue_crops:
                                # 保存裁剪图像
                                crop_tongue_path = self.save_crop_tongue(crop_image)
                                
                                # 保存原始帧
                                original_path = self.save_original_frame(frame, crop_tongue_path)
                                
                                # 标记已发送第一帧图像
                                self.first_image_sent = True
                                
                                # 发送两个路径信号
                                self.crop_tongue_saved_path.emit(crop_tongue_path)
                                self.original_frame_saved_path.emit(original_path, crop_tongue_path)
                                
                                self.last_save_time = current_time
                                self.tongue_crop_count += 1
                                
                                print("已发送第一帧图像进行舌诊分析")
                        # 对后续检测到的舌头图像的处理 (只有在未发送第一帧时才发送)
                        elif detected and self.first_image_sent:
                            # 继续检测但不再发送信号
                            pass
                        else:
                            # 如果没有检测到舌头，切换回预览模式
                            self.working_mode = self.MODE_PREVIEW
                
            else:
                # 队列为空时短暂休眠
                time.sleep(0.01)

        
    def detect_tongue(self, frame):
        """舌头检测函数，使用YOLO模型"""
        if not self.tongue_detection_enabled:
            return False, None, 0, None
        
        try:
            # 如果输入是OpenCV格式(BGR)，转换为RGB以匹配PIL的期望格式
            if isinstance(frame, np.ndarray):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 可选：转换为PIL图像以确保格式完全匹配
                # frame_pil = Image.fromarray(frame_rgb)
                detected, bbox, conf, crop_img = self.tongue_model.detect_single_image(frame_rgb,crop=True)
            else:
                # 如果已经是PIL格式，直接传入
                detected, bbox, conf, crop_img = self.tongue_model.detect_single_image(frame, crop=True)
            
            print(f"载入detect_tongue函数，检测结果为{detected}")
            return detected, bbox, conf, crop_img
        except Exception as e:
            print(f"舌头检测出错: {str(e)}")
            return False, None, 0, None

    def save_crop_tongue(self, crop_image):
        """保存裁剪的舌头图像并返回路径"""
        user_dir = os.path.join(self.save_dir, "tongue_crops")
        os.makedirs(user_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        crop_path = os.path.join(user_dir, f"crop_{timestamp}.jpg")
        # 判断图像类型并保存
        if isinstance(crop_image, np.ndarray):  # OpenCV格式
            cv2.imwrite(crop_path, crop_image)
            # cv2.imwrite(crop_path, crop_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif hasattr(crop_image, 'save'):      # PIL格式
            crop_image.save(crop_path)
            # crop_image.save(crop_path, format='JPEG', quality=95)
        else:
            print(f"无法识别的图像类型: {type(crop_image)}")
            return None
        
        
        
        return crop_path
    
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
        
    def pause(self):
        """暂停线程运行"""
        self.running = False
        # 释放摄像头资源
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        print("摄像头线程已暂停")

    def resume(self):
        """恢复线程运行"""
        if self.isRunning():
            print("摄像头线程已经在运行")
            return  # 如果线程已经在运行，不做任何事
        
        self.running = True
        self.cap = None  # 确保摄像头被重新初始化
        self.start()  # 重新启动线程
        print("摄像头线程已恢复")

    def stop(self):
        """停止线程"""
        self.running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=1.0)
        self.wait()

    def set_diagnosis_completed(self, completed):
        """设置舌诊是否已完成"""
        self.diagnosis_completed = completed
        if completed:
            print("舌诊已完成，舌头检测将停止")
            self.tongue_detection_enabled = False  # 自动禁用舌头检测
        else:
            # 如果重新开始诊断，重置发送图像标志
            self.first_image_sent = False

    def save_original_frame(self, frame, crop_image_path):
        """保存包含舌头的原始帧"""
        # 从裁剪图路径提取基本名称
        base_dir = os.path.dirname(crop_image_path)
        base_name = os.path.basename(crop_image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # 创建原始帧的文件名
        original_path = os.path.join(base_dir, f"{name_without_ext}_original.jpg")
        
        # 保存原始帧
        cv2.imwrite(original_path, frame)
        print(f"原始帧保存到: {original_path}")
        
        return original_path

    def set_face_detection_enabled(self, enabled):
        """启用或禁用面诊检测"""
        self.face_detection_enabled = enabled
        if enabled:
            self.face_diagnosed = False
            self.face_image_count = 0
            print("面诊检测已启用")