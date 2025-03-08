# camera_thread.py

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition
import cv2
import time
from PyQt5.QtMultimedia import QSound
import os
from queue import Queue
from threading import Thread
import numpy as np

class CameraThread(QThread):
    frame_received = pyqtSignal(object)  # 传递OpenCV帧
    snapshot_saved = pyqtSignal(str)      # 传递保存的快照路径
    face_detected = pyqtSignal(bool)      # 传递人脸检测结果
    tongue_detected = pyqtSignal(bool, np.ndarray)  # 传递舌象检测结果
    guidance_message = pyqtSignal(str)    # 发送引导消息信号
    tongue_diagnosis_ready = pyqtSignal(object)  # 新增信号

    # 添加工作模式常量
    MODE_PREVIEW = 0  # 仅预览模式，不保存图像
    MODE_CAPTURE = 1  # 拍摄模式，定期保存图像并分析

    def __init__(self, yolo_model, camera_index=0, snapshot_interval=5, save_folder='snapshots'):
        super().__init__()
        self.snapshot_interval = snapshot_interval
        self.save_folder = save_folder
        self.running = True
        self.camera_index = camera_index
        self.cap = None  # 延迟初始化摄像头
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.yolo_model = yolo_model
        
        # 图像处理相关设置
        self.frame_queue = Queue(maxsize=30)  # 最多缓存30帧
        self.processing_enabled = False  # 是否启用处理
        self.save_enabled = True  # 是否保存图像
        self.frames_to_skip = 10  # 每处理一帧，跳过多少帧
        self.frame_count = 0
        self.processor_thread = None
        self.last_save_time = 0
        
        # 舌头检测相关（预留）
        self.tongue_detection_enabled = False
        self.has_tongue = False
        self.guidance_interval = 3  # 提示间隔，秒

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
            tongue_detected = False

            # 执行舌头检测
            if self.tongue_detection_enabled:
                try:
                    tongue_detected, confidence, bbox = self.detect_tongue(frame)
                except Exception as e:
                    print(f"舌头检测出错: {str(e)}")
                    tongue_detected = False

                # 动态模式切换
                if tongue_detected:
                    # 进入拍摄诊断模式
                    self.working_mode = self.MODE_CAPTURE
                    last_capture_time = current_time
                    
                    # 发送完整帧用于诊断
                    self.tongue_diagnosis_ready.emit(frame.copy())
                else:
                    # 超过拍摄持续时间后切回预览模式
                    if current_time - last_capture_time > capture_duration:
                        self.working_mode = self.MODE_PREVIEW

            # 模式处理逻辑
            if self.working_mode == self.MODE_PREVIEW:
                # 预览模式：降低帧率显示缩放图像
                if current_time - self.last_preview_time >= self.preview_interval:
                    preview_frame = cv2.resize(frame, (0,0), fx=self.preview_scale, fy=self.preview_scale)
                    self.frame_received.emit(preview_frame)
                    self.last_preview_time = current_time

                # 显示引导提示
                if self.tongue_detection_enabled and not tongue_detected:
                    if current_time - last_guidance_time > self.guidance_interval:
                        self.guidance_message.emit("请伸出舌头进行检测")
                        last_guidance_time = current_time

            elif self.working_mode == self.MODE_CAPTURE:
                # 拍摄模式：发送原始帧并保存
                self.frame_received.emit(frame)
                
                # 定期保存图像（示例：每秒1张）
                if current_time - self.last_save_time >= 1.0:
                    self.save_snapshot(frame)
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
                    if self.tongue_detection_enabled:
                        self.has_tongue = self.detect_tongue(frame)
                        self.tongue_detected.emit(self.has_tongue, frame.copy())
                    
                    # 如果启用了保存且达到保存间隔，保存图像
                    current_time = time.time()
                    if self.save_enabled and (current_time - self.last_save_time >= self.snapshot_interval):
                        snapshot_path = self.save_snapshot(frame)
                        self.snapshot_saved.emit(snapshot_path)
                        self.last_save_time = current_time
                
            else:
                # 队列为空时短暂休眠
                time.sleep(0.01)

    def detect_tongue(self, frame):
        """舌头检测预处理接口"""
        # 这里是舌头检测的预留接口
        # 后续可以接入实际的舌头检测算法
        # return False
        return True, 0.9, (50, 50, 100, 100)  # (detected, confidence, bbox)
    

    def detect_face(self, frame):
        """人脸检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return len(faces) > 0

    def save_snapshot(self, frame):
        """保存图像快照"""
        # 使用时间戳生成唯一的文件名
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        snapshot_path = os.path.join(self.save_folder, f"snapshot_{timestamp}.jpg")

        # 确保保存目录存在
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # 保存快照
        cv2.imwrite(snapshot_path, frame)
        return snapshot_path

    def set_save_enabled(self, enabled):
        """设置是否保存图像"""
        self.save_enabled = enabled
        
    def set_tongue_detection_enabled(self, enabled):
        """设置是否启用舌头检测"""
        self.tongue_detection_enabled = enabled
        
    def set_frames_to_skip(self, count):
        """设置跳过的帧数"""
        if count > 0:
            self.frames_to_skip = count
            
    def set_snapshot_interval(self, interval):
        """设置截图间隔（秒）"""
        if interval > 0:
            self.snapshot_interval = interval

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
