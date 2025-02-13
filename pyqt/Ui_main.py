# Ui_main.py
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QComboBox,
    QLabel, QTextBrowser, QInputDialog, QMessageBox, QDialog, QLineEdit,
    QFormLayout, QDialogButtonBox, QVBoxLayout, QHBoxLayout
)
import sys
from PyQt5.QtWidgets import QWidget
from finger_detect import save_finger_pulse
from wrist_detect import save_wrist_pulse
import os
import threading
from wave import finger_PlotWidget
import serial
import serial.tools.list_ports
import json
from datetime import datetime
from wrist_thread import WristDataThread
from finger_thread import FingerDataThread
from camera_thread import CameraThread

class AddUserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加新用户")
        self.layout = QFormLayout(self)

        # 用户名输入
        self.username_input = QLineEdit(self)
        self.layout.addRow("用户名:", self.username_input)

        # 可选信息输入
        self.age_input = QLineEdit(self)
        self.layout.addRow("年龄:", self.age_input)

        self.gender_input = QLineEdit(self)
        self.layout.addRow("性别:", self.gender_input)

        self.medical_history_input = QLineEdit(self)
        self.layout.addRow("病史:", self.medical_history_input)

        # 对话框按钮
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addRow(self.buttons)

    def get_data(self):
        return {
            "username": self.username_input.text().strip(),
            "age": self.age_input.text().strip(),
            "gender": self.gender_input.text().strip(),
            "medical_history": self.medical_history_input.text().strip(),
            "creation_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

class Ui_MainWindow(QWidget):
    def setupUi(self, MainWindow):
        self.isCapture = False
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 1200)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 创建主垂直布局
        self.main_layout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # 创建一个水平布局来包含控制按钮和串口选择
        self.control_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.control_layout)

        # 开始检测按钮
        self.start_b = QtWidgets.QPushButton("开始检测", self.centralwidget)
        self.control_layout.addWidget(self.start_b)
        self.start_b.clicked.connect(self.start_all_sensor)

        # 视频窗口标签
        self.video_l = QtWidgets.QLabel("视频窗口", self.centralwidget)
        self.video_l.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.video_1.setFixedSize(800, 600)  # 设置舌像窗口标签大小为800x600
        self.video_l.setAlignment(QtCore.Qt.AlignCenter)
    
        self.control_layout.addWidget(self.video_l)
        # self.control_layout.setStretchFactor(self.video_1, 2)

        # # 截图按钮
        # self.cut_b = QtWidgets.QPushButton("截图", self.centralwidget)
        # self.control_layout.addWidget(self.cut_b)
        # self.cut_b.clicked.connect(self.capture)

        # 舌像窗口标签
        self.screenshot_l = QtWidgets.QLabel("舌像窗口", self.centralwidget)
        self.screenshot_l.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.screenshot_l.setAlignment(QtCore.Qt.AlignCenter)
        # self.screenshot_l.setFixedSize(800, 600)  # 设置舌像窗口标签大小为800x600
        self.control_layout.addWidget(self.screenshot_l)

        # 面像窗口标签
        self.face_l = QtWidgets.QLabel("面像窗口", self.centralwidget)
        self.face_l.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.face_l.setFixedSize(800, 600)  # 设置舌像窗口标签大小为800x600
        self.face_l.setAlignment(QtCore.Qt.AlignCenter)
        
        self.control_layout.addWidget(self.face_l)

        # 串口选择控件
        # 手腕串口
        self.wrist_serial_label = QtWidgets.QLabel("选择手腕串口:", self.centralwidget)
        self.control_layout.addWidget(self.wrist_serial_label)

        self.wrist_serial_combo = QComboBox(self.centralwidget)
        self.control_layout.addWidget(self.wrist_serial_combo)

        self.confirm_wrist_serial_b = QtWidgets.QPushButton("确认手腕串口", self.centralwidget)
        self.control_layout.addWidget(self.confirm_wrist_serial_b)
        self.confirm_wrist_serial_b.clicked.connect(self.confirm_wrist_serial_selection)

        # 指夹串口
        self.finger_serial_label = QtWidgets.QLabel("选择指夹串口:", self.centralwidget)
        self.control_layout.addWidget(self.finger_serial_label)

        self.finger_serial_combo = QComboBox(self.centralwidget)
        self.control_layout.addWidget(self.finger_serial_combo)

        self.confirm_finger_serial_b = QtWidgets.QPushButton("确认指夹串口", self.centralwidget)
        self.control_layout.addWidget(self.confirm_finger_serial_b)
        self.confirm_finger_serial_b.clicked.connect(self.confirm_finger_serial_selection)
        
        # 用户选择控件
        self.user_label = QtWidgets.QLabel("选择用户:", self.centralwidget)
        self.control_layout.addWidget(self.user_label)

        self.user_combo = QComboBox(self.centralwidget)
        self.control_layout.addWidget(self.user_combo)

        self.add_user_b = QtWidgets.QPushButton("添加用户", self.centralwidget)
        self.control_layout.addWidget(self.add_user_b)
        self.add_user_b.clicked.connect(self.add_user)
        
        # 创建 WristPlotWidget 实例并添加到主布局
        # self.plot_widget = WristPlotWidget()
        # self.main_layout.addWidget(self.plot_widget)

        # 添加诊断文本浏览器
        self.diagnosis_tb = QtWidgets.QTextBrowser(self.centralwidget)
        self.diagnosis_tb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.main_layout.addWidget(self.diagnosis_tb)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        # 定义 patient_list_dp
        self.patient_list_dp = os.path.join(os.path.dirname(__file__), 'user_packages')  # 存储用户信息的文件夹
        if not os.path.exists(self.patient_list_dp):
            os.makedirs(self.patient_list_dp)
        
        # 填充串口选择框
        self.populate_serial_ports()
        # 填充用户选择框
        self.populate_users()

        # 初始化串口属性
        self.finger_serial_port = None
        self.wrist_serial_port = None  # 新增手腕串口属性

        # 初始化当前帧
        self.current_frame = None

        # 初始化数据采集线程
        self.wrist_thread = None
        self.finger_thread = None
        self.camera_thread = None

    def populate_serial_ports(self):
        ports = serial.tools.list_ports.comports()
        # Populate wrist serial ports
        self.wrist_serial_combo.clear()
        for port in ports:
            self.wrist_serial_combo.addItem(port.device)
        if ports:
            self.wrist_serial_combo.setCurrentIndex(0)
        else:
            self.wrist_serial_combo.addItem("没有可用的串口")

        # Populate finger serial ports
        self.finger_serial_combo.clear()
        for port in ports:
            self.finger_serial_combo.addItem(port.device)
        if ports:
            self.finger_serial_combo.setCurrentIndex(0)
        else:
            self.finger_serial_combo.addItem("没有可用的串口")
        
    def populate_users(self):
        self.user_combo.clear()
        for item in os.listdir(self.patient_list_dp):
            item_path = os.path.join(self.patient_list_dp, item)
            if os.path.isdir(item_path):
                self.user_combo.addItem(item)
        if self.user_combo.count() == 0:
            self.user_combo.addItem("无用户，请添加")

    def add_user(self):
        dialog = AddUserDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            user_data = dialog.get_data()
            username = user_data["username"]
            if not username:
                QMessageBox.warning(self, '无效输入', '用户名不能为空。')
                return
            # Validate the user ID (e.g., no special characters)
            if not username.isalnum():
                QMessageBox.warning(self, '无效ID', '用户名只能包含字母和数字。')
                return
            user_path = os.path.join(self.patient_list_dp, username)
            if os.path.exists(user_path):
                QMessageBox.warning(self, '重复ID', '该用户名已存在。')
                return
            os.makedirs(user_path)
            # Save user information to a JSON file
            user_info = {
                "username": username,
                "age": user_data["age"],
                "gender": user_data["gender"],
                "medical_history": user_data["medical_history"],
                "creation_time": user_data["creation_time"]
            }
            with open(os.path.join(user_path, "user_info.json"), "w", encoding='utf-8') as f:
                json.dump(user_info, f, ensure_ascii=False, indent=4)
            self.user_combo.addItem(username)
            self.user_combo.setCurrentText(username)
            print(f"已添加新用户: {username}")

    def confirm_wrist_serial_selection(self):
        selected_port = self.wrist_serial_combo.currentText()
        if selected_port == "没有可用的串口":
            print("没有可用的手腕串口，无法启动传感器。")
            return
        print(f"选择的手腕串口: {selected_port}")
        # 设置手腕串口
        self.wrist_serial_port = selected_port  # 存储手腕串口
        print(f"手腕串口已设置为 {self.wrist_serial_port}")

    def confirm_finger_serial_selection(self):
        selected_port = self.finger_serial_combo.currentText()
        if selected_port == "没有可用的串口":
            print("没有可用的指夹串口，无法启动传感器。")
            return
        print(f"选择的指夹串口: {selected_port}")
        # 设置指夹串口
        self.finger_serial_port = selected_port  # Store the selected finger serial port
        print(f"指夹串口已设置为 {self.finger_serial_port}")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Patient Data Collector"))
        self.start_b.setText(_translate("MainWindow", "开始检测"))
        self.video_l.setText(_translate("MainWindow", "视频窗口"))
        # self.cut_b.setText(_translate("MainWindow", "截图"))
        self.screenshot_l.setText(_translate("MainWindow", "舌像窗口"))
        self.face_l.setText(_translate("MainWindow", "面像窗口"))
        self.wrist_serial_label.setText(_translate("MainWindow", "选择手腕串口:"))
        self.finger_serial_label.setText(_translate("MainWindow", "选择指夹串口:"))
        self.confirm_wrist_serial_b.setText(_translate("MainWindow", "确认手腕串口"))
        self.confirm_finger_serial_b.setText(_translate("MainWindow", "确认指夹串口"))
        self.user_label.setText(_translate("MainWindow", "选择用户:"))
        self.add_user_b.setText(_translate("MainWindow", "添加用户"))

    # 对应开始检测的按钮
    def start_all_sensor(self):
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QtWidgets.QMessageBox.warning(self, '无用户', '请先添加用户。')
            return
        user_folder = os.path.join(self.patient_list_dp, selected_user)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        self.patient_id = selected_user  # 使用用户名作为 patient_id

        # 启动数据采集线程
        # 1. 指夹传感器
        if self.finger_serial_port and self.finger_serial_port != "没有可用的串口":
            if self.finger_thread is None:
                self.finger_thread = FingerDataThread(
                    serial_port=self.finger_serial_port,
                    baudrate=115200,
                )
                # self.finger_thread.data_received.connect(self.handle_finger_data)
                self.finger_thread.csv_filename = os.path.join(user_folder, "finger_pulse.csv")
                self.finger_thread.start()
                print("指夹数据采集线程已启动。")
            else:
                print("指夹数据采集线程已在运行。")
        else:
            print("指夹串口未选择或初始化失败。")

        # 2. 手腕传感器
        if self.wrist_serial_port and self.wrist_serial_port != "没有可用的串口":
            if self.wrist_thread is None:
                self.wrist_thread = WristDataThread(
                    serial_port=self.wrist_serial_port,
                    baudrate=38400,
                )
                # self.wrist_thread.data_received.connect(self.plot_widget.update_plot)
                self.wrist_thread.csv_filename = os.path.join(user_folder, "wrist_pulse.csv")
                self.wrist_thread.start()
                print("手腕数据采集线程已启动。")
            else:
                print("手腕数据采集线程已在运行。")
        else:
            print("手腕串口未选择或初始化失败。")

        # 3. 摄像头
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'runs', 'detect', 'train', 'weights', 'best.pt')
        model = YOLO(model_path)  # 加载模型
        if self.camera_thread is None:
            self.camera_thread = CameraThread(
                snapshot_interval=5,  # 每5秒保存一次截图
                save_folder=user_folder,
                yolo_model=model
            )
            self.camera_thread.frame_received.connect(self.display_camera_frame)
            self.camera_thread.snapshot_saved.connect(self.handle_snapshot_saved)
            self.camera_thread.start()
            print("摄像头线程已启动。")
        else:
            print("摄像头线程已在运行。")

    def handle_finger_data(self, pulse_value):
        # 如果需要实时显示指夹数据，可以实现类似于手腕数据的绘图
        # 这里简单打印数据
        print(f"指夹传感器数据: {pulse_value}")
        # TODO: 可以添加一个指夹传感器的PlotWidget来实时显示数据

    def display_camera_frame(self, frame):
        """
        显示摄像头画面，并根据flip_mode参数翻转图像。
        
        :param frame: OpenCV捕获的图像帧
        :param flip_mode: 翻转模式，可选值为0（垂直翻转）、1（水平翻转）、-1（同时水平垂直翻转），默认为None（不翻转）
        """
        # 根据flip_mode参数翻转图像
        # if flip_mode is not None:
        frame = cv2.flip(frame, 0)
        
        # 将OpenCV图像转换为Qt图像并显示在video_l标签中
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(rgb_image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_l.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.video_l.setPixmap(scaled_pixmap)
        annotated_frame, diagnosis = tongue_diagnosis(frame)
        self.show_diagnosis(diagnosis)

    def handle_snapshot_saved(self, snapshot_path):
        # 显示截图在face_l标签中
        pixmap = QtGui.QPixmap(snapshot_path)
        transform = QtGui.QTransform()
        transform = transform.scale(1, -1)  # 垂直翻转
        transformed_pixmap = pixmap.transformed(transform)
        scaled_pixmap = transformed_pixmap.scaled(self.face_l.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.face_l.setPixmap(scaled_pixmap)
        print(f"已显示截图: {snapshot_path}")

    # 对应截图的按钮
    def capture(self):
        if not hasattr(self, 'patient_id') or self.patient_id == -1:
            QtWidgets.QMessageBox.warning(self, '无用户', '请先选择或添加用户。')
            return
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            img_fp = os.path.join(self.patient_list_dp, str(self.patient_id), "tongue.jpg")
            # 保存原始帧
            cv2.imwrite(img_fp, self.current_frame)
            print("保存完毕")
            self.isCapture = True  # 设置标志为已截图
            # 显示截图在截图 QLabel 中
            self.update_screenshot(self.current_frame)
        else:
            QtWidgets.QMessageBox.warning(self, '无帧', '没有可用的帧进行保存。')
            print("没有可用的帧进行保存。")

    def update_screenshot(self, frame):
        # 在这里对图像进行处理，例如打印图像的尺寸
        annotated_frame, diagnosis = tongue_diagnosis(frame)
        # 改图片标签
        img = QtGui.QImage(annotated_frame.data, annotated_frame.shape[1], annotated_frame.shape[0], QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(img)
        scaled_pixmap = pixmap.scaled(self.screenshot_l.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.screenshot_l.setPixmap(scaled_pixmap)
        # 更改诊断
        self.show_diagnosis(diagnosis)

    def show_diagnosis(self, diag):
        self.diag = diag
        self.index = 0
        self.diagnosis_tb.clear()  # 清空文本浏览器
        self.timer = self.startTimer(100)  # 每 100 毫秒触发一次定时器事件

    def timerEvent(self, event):
        if self.index < len(self.diag):
            self.diagnosis_tb.insertPlainText(self.diag[self.index])  # 插入一个字符
            self.index += 1
        else:
            self.killTimer(self.timer)  # 文本显示完毕，停止定时器

    def closeEvent(self, event):
        # 当窗口关闭时，确保所有线程都被正确停止
        if self.wrist_thread is not None:
            self.wrist_thread.stop()
        if self.finger_thread is not None:
            self.finger_thread.stop()
        if self.camera_thread is not None:
            self.camera_thread.stop()
        event.accept()


def tongue_diagnosis(img):
    class_labels = {
        0: "您的舌质呈现粉红色，这通常与健康的舌象相符，表明您的身体状况良好，气血充足。粉红舌通常反映出良好的生理状态，然而，如果舌质偏红，则可能提示体内存在热症，需警惕潜在的炎症或感染情况。建议定期关注身体其他症状，保持健康的生活方式。",
        1: "您的舌苔薄白，通常表明体内没有明显的病理变化，可能处于健康状态。然而，薄白舌也可能提示气血不足或体内寒气较重，建议注意饮食调理，适当增加营养摄入，保持身体温暖，避免寒凉食物的过量摄入。",
        2: "您的舌苔厚白，这可能指示体内存在寒湿或痰湿，通常与消化系统功能障碍有关。厚白舌常见于脾胃虚弱、消化不良等情况，建议您关注饮食习惯，避免油腻和生冷食物，同时可以考虑适当的中药调理，以增强脾胃功能。",
        3: "您的舌苔厚黄，这通常表示体内有湿热，可能伴随发热、口渴、便秘等症状。厚黄舌常见于感染、炎症或消化系统疾病。建议您保持充足的水分摄入，避免辛辣刺激食物，同时可以考虑咨询专业医生进行进一步检查和调理。",
        4: "您的舌苔灰黑，这是一种较为严重的病理变化，可能与严重的感染、长期疾病、药物中毒或内脏器官的严重病变有关。灰黑舌通常提示体内存在较大的病理变化，建议您尽快就医，进行详细检查，以便及时发现并处理潜在的健康问题。"
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'runs', 'detect', 'train', 'weights', 'best.pt')
    # model = YOLO(model_path)  # 加载模型
    # model_path = './runs/detect/train/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"模型文件不存在：{model_path}")
        return img, "模型文件不存在，请检查模型路径"

    model = YOLO(model_path)  # 加载模型
    results = model(img)
    if not results:
        print("您的舌质呈现粉红色，这通常与健康的舌象相符，表明您的身体状况良好，气血充足。粉红舌通常反映出良好的生理状态，然而，如果舌质偏红，则可能提示体内存在热症，需警惕潜在的炎症或感染情况。建议定期关注身体其他症状，保持健康的生活方式。")
        return img, "未检测到舌像，请重新拍照"

    annotated_frame = results[0].plot()
    diagnosis = "没有发现舌像，请重新拍照"
    for result in results:
        class_ids = result.boxes.cls.numpy()  # 获取类别索引数组
        for class_id in class_ids:
            diagnosis = class_labels.get(int(class_id), "未知类别")
   
    print("您的舌质呈现粉红色，这通常与健康的舌象相符，表明您的身体状况良好，气血充足。粉红舌通常反映出良好的生理状态，然而，如果舌质偏红，则可能提示体内存在热症，需警惕潜在的炎症或感染情况。建议定期关注身体其他症状，保持健康的生活方式。")
    return annotated_frame, diagnosis


def find_max_number_in_folders(folder_path):
    max_number = 0  # 初始化最大数字为 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            try:
                number = int(item)
                if number > max_number:
                    max_number = number
            except ValueError:
                pass  # 文件夹名称不是数字，忽略
    print(max_number)
    return max_number

if __name__=="__main__":

    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()     
    ui = Ui_MainWindow()             
    ui.setupUi(MainWindow1)
    MainWindow1.show()
    sys.exit(app.exec_())
