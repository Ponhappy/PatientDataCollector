# camera_thread.py

from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import time
from PyQt5.QtMultimedia import QSound
from tongue_detection import detect_tongue

class CameraThread(QThread):
    frame_received = pyqtSignal(object)  # 传递OpenCV帧
    snapshot_saved = pyqtSignal(str)      # 传递保存的快照路径
    face_detected = pyqtSignal(bool)      # 传递人脸检测结果
    tongue_detected = pyqtSignal(bool)    # 传递舌象检测结果

    def __init__(self, yolo_model, snapshot_interval=5, save_folder='snapshots'):
        super().__init__()
        self.snapshot_interval = snapshot_interval
        self.save_folder = save_folder
        self.running = True
        self.cap = cv2.VideoCapture(0)  # 0为默认摄像头
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.yolo_model = yolo_model

    def run(self):
        last_snapshot_time = time.time()
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 发送帧给主线程显示
                self.frame_received.emit(frame)

                # 人脸检测
                face_present = self.detect_face(frame)
                self.face_detected.emit(face_present)

                if not face_present:
                    # 发出声音提示用户靠近或远离
                    QSound.play('alert_sound.wav')  # 确保有 'alert_sound.wav'

                # 检查是否需要保存快照
                current_time = time.time()
                if current_time - last_snapshot_time >= self.snapshot_interval:
                    snapshot_path = self.save_snapshot(frame)
                    self.snapshot_saved.emit(snapshot_path)
                    last_snapshot_time = current_time

                # 舌象检测
                tongue_present, _ = detect_tongue(frame, self.yolo_model)
                self.tongue_detected.emit(tongue_present)
                if not tongue_present:
                    QSound.play('alert_tongue.wav')  # 确保有 'alert_tongue.wav'

            time.sleep(0.03)  # 约30帧每秒

        self.cap.release()

    def detect_face(self, frame):
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
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        snapshot_path = f"{self.save_folder}/snapshot_{timestamp}.jpg"
        cv2.imwrite(snapshot_path, frame)
        return snapshot_path

    def stop(self):
        self.running = False
        self.wait()
