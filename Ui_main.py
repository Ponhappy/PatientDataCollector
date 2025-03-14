from ultralytics import YOLO
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QComboBox,
    QLabel, QTextBrowser, QInputDialog, QMessageBox, QDialog, QLineEdit,
    QFormLayout, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QWidget,
    QStatusBar, QToolButton, QGroupBox, QTabWidget, QScrollArea, QGridLayout,
    QStyleFactory
)
import sys
from PyQt5.QtWidgets import QWidget
import os
import threading
import serial
import serial.tools.list_ports
import json
from datetime import datetime
from wrist_thread import WristDataThread
from finger_thread import FingerDataThread
from camera_thread import CameraThread
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QImage
from tongue_diagnose_model.tongue_diagnose import tongue_diagnose_sum
# import resources_rc  # 资源文件

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel


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

class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能舌象采集分析系统")
        self.setMinimumSize(1200, 800)  # 设置最小窗口大小
        
        # 初始化患者数据存储路径
        self.patient_list_dp = os.path.join(os.path.expanduser("~"), "patient_data")  # 默认存储在用户目录下
        if not os.path.exists(self.patient_list_dp):
            os.makedirs(self.patient_list_dp)  # 自动创建目录
        
        # 添加舌诊处理标志
        self.tongue_diagnosed = False
        
        # 创建并设置中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # 创建左侧控制面板和右侧显示区域
        self.setup_control_panel()
        self.setup_display_area()
        
        # 创建底部状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪")
        
        # 加载并应用样式表
        self.apply_stylesheet()
        
        # 初始化变量
        self.camera_thread = None
        self.wrist_thread = None
        self.finger_thread = None
        self.camera_index = 0
        # 初始化串口属性
        self.finger_serial_port = None
        self.wrist_serial_port = None
        
        # 填充选项框
        self.populate_cameras()
        self.populate_serial_ports()
        self.populate_users()

    def setup_control_panel(self):
        """设置左侧控制面板"""
        # 创建左侧面板容器
        self.control_panel = QWidget()
        self.control_panel.setObjectName("controlPanel")
        self.control_panel.setMaximumWidth(350)
        self.control_panel.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        
        # 控制面板布局
        control_layout = QtWidgets.QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(15)
        
        # 创建用户选择组
        user_group = self.create_group_box("用户选择")
        user_layout = QtWidgets.QVBoxLayout(user_group)
        
        # 用户选择控件
        self.user_combo = QtWidgets.QComboBox()
        self.user_combo.setObjectName("userCombo")
        self.add_user_btn = QtWidgets.QPushButton("添加用户")
        self.add_user_btn.setIcon(QtGui.QIcon(":/icons/add_user.png"))
        self.add_user_btn.clicked.connect(self.add_user)
        
        user_layout.addWidget(QtWidgets.QLabel("选择用户:"))
        user_layout.addWidget(self.user_combo)
        user_layout.addWidget(self.add_user_btn)
        
        # 创建设备配置组
        device_group = self.create_group_box("设备配置")
        device_layout = QtWidgets.QGridLayout(device_group)
        
        # 摄像头选择
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.setObjectName("deviceCombo")
        self.confirm_camera_btn = QtWidgets.QPushButton("确认")
        self.confirm_camera_btn.clicked.connect(self.confirm_camera_selection)
        device_layout.addWidget(QtWidgets.QLabel("摄像头:"), 0, 0)
        device_layout.addWidget(self.camera_combo, 0, 1)
        device_layout.addWidget(self.confirm_camera_btn, 0, 2)
        
        # 腕带串口
        self.wrist_serial_label = QtWidgets.QLabel("选择腕带串口:")
        self.wrist_serial_combo = QtWidgets.QComboBox()
        self.wrist_serial_combo.setObjectName("deviceCombo")
        self.confirm_wrist_btn = QtWidgets.QPushButton("确认")
        self.confirm_wrist_btn.clicked.connect(self.confirm_wrist_selection)
        device_layout.addWidget(self.wrist_serial_label, 1, 0)
        device_layout.addWidget(self.wrist_serial_combo, 1, 1)
        device_layout.addWidget(self.confirm_wrist_btn, 1, 2)
        
        # 指夹串口
        self.finger_serial_label = QtWidgets.QLabel("选择指夹串口:")
        self.finger_serial_combo = QtWidgets.QComboBox()
        self.finger_serial_combo.setObjectName("deviceCombo")
        self.confirm_finger_btn = QtWidgets.QPushButton("确认")
        self.confirm_finger_btn.clicked.connect(self.confirm_finger_selection)
        device_layout.addWidget(self.finger_serial_label, 2, 0)
        device_layout.addWidget(self.finger_serial_combo, 2, 1)
        device_layout.addWidget(self.confirm_finger_btn, 2, 2)
        
        # 创建操作控制组
        operation_group = self.create_group_box("操作控制")
        operation_layout = QtWidgets.QVBoxLayout(operation_group)
        
        # 创建操作按钮
        self.start_camera_btn = self.create_action_button("启动摄像头", ":/icons/camera.png")
        self.start_camera_btn.clicked.connect(self.start_camera_only)
        
        self.start_sensors_btn = self.create_action_button("开始脉象采集", ":/icons/sensor.png")
        self.start_sensors_btn.clicked.connect(self.start_sensors_only)
        
        self.stop_sensors_btn = self.create_action_button("停止脉象采集", ":/icons/stop.png")
        self.stop_sensors_btn.clicked.connect(self.stop_sensors)
        

        self.refresh_devices_btn = self.create_action_button("刷新设备列表", ":/icons/refresh.png")
        self.refresh_devices_btn.clicked.connect(self.refresh_devices)
        
        # 添加按钮到操作布局
        operation_layout.addWidget(self.start_camera_btn)
        operation_layout.addWidget(self.start_sensors_btn)
        operation_layout.addWidget(self.stop_sensors_btn)
        operation_layout.addWidget(self.refresh_devices_btn)
        
        # 将所有组添加到控制面板
        control_layout.addWidget(user_group)
        control_layout.addWidget(device_group)
        control_layout.addWidget(operation_group)
        control_layout.addStretch()
        
        # 将控制面板添加到主布局
        self.main_layout.addWidget(self.control_panel)

    def setup_display_area(self):
        """设置右侧显示区域"""
        # 创建右侧显示区域容器
        self.display_area = QWidget()
        self.display_area.setObjectName("displayArea")
        
        # 显示区域布局
        display_layout = QtWidgets.QVBoxLayout(self.display_area)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(10)
        
        # 创建上部相机视图和下部分析结果区域的分割器
        self.main_splitter = QtWidgets.QSplitter(Qt.Vertical)
        
        # 创建相机视图区域
        camera_view_container = QWidget()
        camera_view_layout = QtWidgets.QHBoxLayout(camera_view_container)
        camera_view_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建画面显示区域
        self.video_display = QtWidgets.QLabel("等待摄像头启动...")
        self.video_display.setObjectName("videoDisplay")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("background-color: #2c3e50; color: white;")
        
        camera_view_layout.addWidget(self.video_display)
        
        # 创建分析结果区域
        self.analysis_container = QtWidgets.QTabWidget()
        self.analysis_container.setObjectName("analysisTab")
        
        # 舌象分析标签页
        tongue_tab = QtWidgets.QWidget()
        tongue_layout = QtWidgets.QHBoxLayout(tongue_tab)
        
        # 左侧舌象图像显示
        self.tongue_image = QtWidgets.QLabel("舌象图像将在此显示")
        self.tongue_image.setObjectName("analysisImage")
        self.tongue_image.setAlignment(Qt.AlignCenter)
        self.tongue_image.setStyleSheet("background-color: #34495e; color: white;")
        
        # 右侧舌象分析结果
        self.tongue_results = QtWidgets.QTextBrowser()
        self.tongue_results.setObjectName("analysisText")
        
        tongue_layout.addWidget(self.tongue_image, 1)
        tongue_layout.addWidget(self.tongue_results, 1)
        
        # 面象分析标签页
        face_tab = QtWidgets.QWidget()
        face_layout = QtWidgets.QHBoxLayout(face_tab)
        
        # 左侧面像图像显示
        self.face_image = QtWidgets.QLabel("面像图像将在此显示")
        self.face_image.setObjectName("analysisImage")
        self.face_image.setAlignment(Qt.AlignCenter)
        self.face_image.setStyleSheet("background-color: #34495e; color: white;")
        
        # 右侧面象分析结果
        self.face_results = QtWidgets.QTextBrowser()
        self.face_results.setObjectName("analysisText")
        
        face_layout.addWidget(self.face_image, 1)
        face_layout.addWidget(self.face_results, 1)
        
        # 脉象分析标签页
        sensor_tab = QtWidgets.QWidget()
        sensor_layout = QtWidgets.QVBoxLayout(sensor_tab)
        
        # 脉象分析显示区域
        self.sensor_data = QtWidgets.QTextBrowser()
        self.sensor_data.setObjectName("sensorData")
        
        sensor_layout.addWidget(self.sensor_data)
        
        # 诊断结果标签页
        diagnosis_tab = QtWidgets.QWidget()
        diagnosis_layout = QtWidgets.QVBoxLayout(diagnosis_tab)
        
        # 综合诊断结果显示
        self.diagnosis_text = QtWidgets.QTextBrowser()
        self.diagnosis_text.setObjectName("diagnosisText")
        
        diagnosis_layout.addWidget(self.diagnosis_text)
        
        # 添加标签页
        self.analysis_container.addTab(tongue_tab, "舌象分析")
        self.analysis_container.addTab(face_tab, "面象分析")
        self.analysis_container.addTab(sensor_tab, "脉象分析")
        self.analysis_container.addTab(diagnosis_tab, "综合诊断")
        
        # 将相机视图和分析结果添加到分割器
        self.main_splitter.addWidget(camera_view_container)
        self.main_splitter.addWidget(self.analysis_container)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 2)
        
        # 添加分割器到显示区域
        display_layout.addWidget(self.main_splitter)
        
        # 将显示区域添加到主布局
        self.main_layout.addWidget(self.display_area, 1)  # 1表示拉伸因子，使显示区域占据更多空间

    def create_group_box(self, title):
        """创建分组框"""
        group = QtWidgets.QGroupBox(title)
        group.setObjectName("groupBox")
        return group

    def create_action_button(self, text, icon_path=None):
        """创建操作按钮"""
        btn = QtWidgets.QPushButton(text)
        btn.setObjectName("actionButton")
        if icon_path:
            btn.setIcon(QtGui.QIcon(icon_path))
        return btn

    def apply_stylesheet(self):
        """应用样式表"""
        style = """
        QMainWindow {
            background-color: #f5f5f5;
        }
        
        #controlPanel {
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        
        #displayArea {
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 10px;
            background-color: #ffffff;
        }
        
        #actionButton {
            background-color: #3498db;
            color: white;
            border-radius: 4px;
            padding: 8px;
            font-weight: bold;
            border: none;
        }
        
        #actionButton:hover {
            background-color: #2980b9;
        }
        
        #actionButton:pressed {
            background-color: #1c6ea4;
        }
        
        QComboBox {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            padding: 4px;
            background-color: white;
        }
        
        QComboBox:hover {
            border-color: #3498db;
        }
        
        QTextBrowser {
            border: 1px solid #d0d0d0;
            border-radius: 4px;
            background-color: white;
        }
        
        #videoDisplay, #analysisImage {
            border-radius: 6px;
            border: 1px solid #d0d0d0;
        }
        
        QTabWidget::pane {
            border: 1px solid #d0d0d0;
            border-radius: 6px;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #f0f0f0;
            border: 1px solid #d0d0d0;
        }
        
        QTabBar::tab:selected {
            background-color: #3498db;
        }
        """
        self.central_widget.setStyleSheet(style)

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
    
    def refresh_devices(self):
        """刷新设备列表（摄像头和串口）"""
        # 刷新摄像头列表
        self.camera_combo.clear()
        self.populate_cameras()
        
        # 刷新串口列表
        self.wrist_serial_combo.clear()
        self.finger_serial_combo.clear()
        self.populate_serial_ports()
        
        # 更新状态栏
        self.status_bar.showMessage("设备列表已刷新")
        
    def populate_cameras(self):
        """填充可用的摄像头列表"""
        self.camera_combo.clear()
        camera_list = []
        
        # 使用更可靠的摄像头检测方式
        for index in range(4):  # 最多检测4个摄像头
            try:
                # 显式指定DSHOW后端并设置超时
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置期望分辨率
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                if cap.isOpened():
                    # 获取摄像头信息
                    camera_name = f"摄像头 {index}"
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    camera_name += f" ({width}x{height})"
                    camera_list.append(camera_name)
                    cap.release()
                else:
                    cap.release()
                    break
            except Exception as e:
                print(f"检测摄像头 {index} 时出错: {str(e)}")
                continue
        
        if camera_list:
            self.camera_combo.addItems(camera_list)
        else:
            self.camera_combo.addItem("未检测到摄像头")
            QMessageBox.warning(self, "摄像头错误", 
                             "未检测到可用摄像头，请检查：\n"
                             "1. 摄像头是否已连接\n"
                             "2. 驱动程序是否正常\n"
                             "3. 其他程序是否占用了摄像头")

    def populate_users(self):
        # 重置舌诊处理标志
        self.tongue_diagnosed = False
        
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

    def confirm_wrist_selection(self):
        selected_port = self.wrist_serial_combo.currentText()
        if selected_port == "没有可用的串口" or not selected_port.startswith("COM"):
            print("没有可用的腕带串口，无法启动传感器。")
            return
        print(f"选择的腕带串口: {selected_port}")
        # 设置腕带串口
        self.wrist_serial_port = selected_port  # 存储腕带串口
        print(f"腕带串口已设置为 {self.wrist_serial_port}")

    def confirm_finger_selection(self):
        selected_port = self.finger_serial_combo.currentText()
        if selected_port == "没有可用的串口" or not selected_port.startswith("COM"):
            print("没有可用的指夹串口，无法启动传感器。")
            return
        print(f"选择的指夹串口: {selected_port}")
        # 设置指夹串口
        self.finger_serial_port = selected_port  # Store the selected finger serial port
        print(f"指夹串口已设置为 {self.finger_serial_port}")

    def confirm_camera_selection(self):
        selected_camera = self.camera_combo.currentText()
        if selected_camera == "没有找到摄像头":
            print("没有可用的摄像头，无法启动视频采集。")
            return
        
        camera_index = int(selected_camera.split()[1])
        self.camera_index = camera_index
        print(f"摄像头已设置为: {camera_index}")

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Patient Data Collector"))
        self.start_b.setText(_translate("MainWindow", "开始采集脉象数据"))
        self.camera_b.setText(_translate("MainWindow", "开始摄像头采集"))
        self.video_display.setText(_translate("MainWindow", "视频窗口"))
        self.screenshot_l.setText(_translate("MainWindow", "舌像窗口"))
        self.face_l.setText(_translate("MainWindow", "面像窗口"))
        self.wrist_serial_label.setText(_translate("MainWindow", "选择腕带串口:"))
        self.finger_serial_label.setText(_translate("MainWindow", "选择指夹串口:"))
        self.confirm_wrist_serial_b.setText(_translate("MainWindow", "确认腕带串口"))
        self.confirm_finger_serial_b.setText(_translate("MainWindow", "确认指夹串口"))
        self.user_label.setText(_translate("MainWindow", "选择用户:"))
        self.add_user_b.setText(_translate("MainWindow", "添加用户"))
        self.stop_sensors_b.setText(_translate("MainWindow", "停止脉象采集"))
        self.stop_camera_b.setText(_translate("MainWindow", "停止摄像头采集"))
        self.camera_label.setText(_translate("MainWindow", "选择摄像头:"))
        self.confirm_camera_b.setText(_translate("MainWindow", "确认摄像头"))
        self.camera_mode_btn.setText(_translate("MainWindow", "开始拍摄"))

    # 仅启动指夹和腕带传感器
    def start_sensors_only(self):
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QtWidgets.QMessageBox.warning(self, '无用户', '请先添加用户。')
            return
        user_folder = os.path.join(self.patient_list_dp, selected_user)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        self.patient_id = selected_user  # 使用用户名作为 patient_id

        # 1. 指夹传感器
        if self.finger_serial_port and self.finger_serial_port != "没有可用的串口":
            if self.finger_thread is None:
                self.finger_thread = FingerDataThread(
                    serial_port=self.finger_serial_port,
                    baudrate=115200,
                )
                self.finger_thread.csv_filename = os.path.join(user_folder, "finger_pulse.csv")
                self.finger_thread.data_received.connect(self.show_sensors_report)
                self.finger_thread.start()
                print("指夹数据采集线程已启动。")
            else:
                print("指夹数据采集线程已在运行。")
        else:
            print("指夹串口未选择或初始化失败。")

        # 2. 腕带传感器
        if self.wrist_serial_port and self.wrist_serial_port != "没有可用的串口":
            if self.wrist_thread is None:
                self.wrist_thread = WristDataThread(
                    serial_port=self.wrist_serial_port,
                    baudrate=38400,
                )
                self.wrist_thread.csv_filename = os.path.join(user_folder, "wrist_pulse.csv")
                self.wrist_thread.data_received.connect(self.show_sensors_report)
                self.wrist_thread.start()
                print("腕带数据采集线程已启动。")
            else:
                print("腕带数据采集线程已在运行。")
        else:
            print("腕带串口未选择或初始化失败。")


    def show_sensors_report(self, report):
        # 将报告内容添加到脉象分析文本浏览器
        # 不包含时间戳版本
        # formatted_report = f"{report}"
        # 包含时间戳版本
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_report = f"[{timestamp}] {report}"
        
        # 在脉象分析标签页中显示
        self.sensor_data.append(formatted_report)
        
        # 同时在诊断结果标签页中也添加摘要信息
        summary = f"[{timestamp}] 收到新的脉象分析报告"
        self.diagnosis_text.append(summary)
        
        # 切换到脉象分析标签页，确保用户能立即看到结果
        # 找到包含传感器标签页的 QTabWidget
        for i in range(self.analysis_container.count()):
            if self.analysis_container.tabText(i) == "脉象分析":
                self.analysis_container.setCurrentIndex(i)
                break
        
        # 更新状态栏信息
        self.status_bar.showMessage("智能脉诊已完成")

    # 仅启动摄像头
    def start_camera_only(self):
        # 重置舌诊处理标志
        self.tongue_diagnosed = False
        
        # 确保摄像头索引有效
        if self.camera_combo.currentText() == "未检测到摄像头":
            QMessageBox.critical(self, "错误", "没有可用的摄像头")
            return
        
        # 从下拉框获取实际摄像头索引
        selected_text = self.camera_combo.currentText()
        try:
            self.camera_index = int(selected_text.split()[1])  # 从"摄像头 0"提取数字
        except:
            QMessageBox.critical(self, "错误", "摄像头选择无效")
            return
        
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QtWidgets.QMessageBox.warning(self, '无用户', '请先添加用户。')
            return
        user_folder = os.path.join(self.patient_list_dp, selected_user)
        if not os.path.exists(user_folder):
            os.makedirs(user_folder)
        self.patient_id = selected_user  # 使用用户名作为 patient_id

        # 启动摄像头线程
        if self.camera_thread is None:
            self.camera_thread = CameraThread(
                save_folder=user_folder,
                crop_tongue_interval=5,
                camera_index=self.camera_index
            )
            self.camera_thread.tongue_detection_enabled = True 
            # 连接信号
            self.camera_thread.frame_received.connect(self.display_camera_frame)
            # self.camera_thread.tongue_detected.connect(self.handle_tongue_detected)
            self.camera_thread.guidance_message.connect(self.show_guidance)
            # self.camera_thread.tongue_diagnosis_ready.connect(self.handle_new_crop_image)
            self.camera_thread.crop_tongue_saved_path.connect(self.handle_new_crop_image)
            self.camera_thread.max_images_reached.connect(self.handle_max_images_reached)
            
            # 配置摄像头线程 
            self.camera_thread.set_frames_to_skip(15)
            self.camera_thread.set_crop_tongue_interval(3)
            
            self.camera_thread.start()
            print(f"摄像头线程已启动，使用摄像头索引: {self.camera_index}")
        else:
            print("摄像头线程已在运行。")

    def handle_finger_data(self, pulse_value):
        # 如果需要实时显示指夹数据，可以实现类似于腕带数据的绘图
        # 这里简单打印数据
        print(f"指夹传感器数据: {pulse_value}")
        # TODO: 可以添加一个指夹传感器的PlotWidget来实时显示数据

    def display_camera_frame(self, frame):
        """显示摄像头帧"""
        try:
            # 将OpenCV的BGR格式转换为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            
            # 转换为QPixmap并显示
            pixmap = QtGui.QPixmap.fromImage(q_img)
            
            # 根据标签大小调整图像大小
            pixmap = pixmap.scaled(self.video_display.width(), self.video_display.height(), 
                                   QtCore.Qt.KeepAspectRatio)
            self.video_display.setPixmap(pixmap)
        except Exception as e:
            print(f"显示摄像头帧出错: {e}")



    def show_diagnosis(self, diag):
        self.diag = diag
        self.index = 0
        self.diagnosis_text.clear()  # 清空文本浏览器
        self.timer = self.startTimer(100)  # 每 100 毫秒触发一次定时器事件

    def timerEvent(self, event):
        if self.index < len(self.diag):
            self.diagnosis_text.insertPlainText(self.diag[self.index])  # 插入一个字符
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

    # 对应原本的开始检测的按钮 - 现在作为综合方法，调用上面两个新方法
    def start_all_sensor(self):
        # 顺序调用两个新方法，同时启动所有传感器
        self.start_sensors_only()
        self.start_camera_only()

    def stop_sensors(self):
        if self.wrist_thread is not None:
            self.wrist_thread.stop()
            self.wrist_thread = None
            print("腕带数据采集线程已停止。")
        if self.finger_thread is not None:
            self.finger_thread.stop()
            self.finger_thread = None
            print("指夹数据采集线程已停止。")

    def stop_camera(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
            print("摄像头线程已停止。")

    def toggle_camera_mode(self):
        """切换摄像头工作模式"""
        if self.camera_thread is None:
            return
        
        if self.camera_thread.working_mode == CameraThread.MODE_PREVIEW:
            # 从预览切换到拍摄模式
            self.camera_thread.set_mode(CameraThread.MODE_CAPTURE)
            self.camera_mode_btn.setText("停止拍摄")
            self.diagnosis_text.append("已开始舌象拍摄和分析")
        else:
            # 从拍摄切换到预览模式
            self.camera_thread.set_mode(CameraThread.MODE_PREVIEW)
            self.camera_mode_btn.setText("开始拍摄")
            self.diagnosis_text.append("已停止拍摄，进入预览模式")

    def toggle_tongue_detection(self, enabled):
        """处理舌头检测开关状态"""
        if self.camera_thread:  # 确保摄像头线程已创建
            self.camera_thread.set_tongue_detection_enabled(enabled)
            status = "启用" if enabled else "禁用"
            self.status_bar.showMessage(f"舌头检测已{status}")
            self.diagnosis_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 舌头检测{status}")
        else:
            self.status_bar.showMessage("请先启动摄像头再操作")


    def show_guidance(self, message):
        """显示检测引导提示"""
        print("调用show_guidance函数")
        
            
        # 更新状态栏
        status_msg = message
        self.status_bar.showMessage(status_msg)
            
            

    def perform_tongue_diagnosis(self, crop_path):
        # 实现舌头诊断逻辑
        # 调用舌头诊断函数
        color_report, coating_report, cancer_report, tongue_annotated, diagnosis, treatment = tongue_diagnose_sum(crop_path)
        print(color_report, coating_report, cancer_report, diagnosis, treatment)
        # self.diagnosis_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 诊断结果：{diagnosis_result}")
        pass

    def handle_new_crop_image(self, crop_path):
        """处理新裁剪的舌头图像"""
        # 如果已经进行过诊断，则忽略后续的图像
        if self.tongue_diagnosed:
            return
        
        if not os.path.exists(crop_path):
            print(f"文件不存在: {crop_path}")
            return
        
        # 显示第一张裁剪图
        if not hasattr(self, 'latest_crop_path'):
            self.display_first_crop(crop_path)
        
        # 执行舌诊分析
        self.perform_tongue_diagnosis(crop_path)
        
        # 标记已完成诊断
        self.tongue_diagnosed = True
        self.status_bar.showMessage("舌诊分析已完成")
        print("舌诊分析已完成，后续图像将不再触发诊断")

    def display_first_crop(self, path):
        """显示第一张裁剪图像"""
        pixmap = QPixmap(path)
        self.tongue_image.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio))
        self.latest_crop_path = path
        
    def handle_max_images_reached(self):
        self.diagnosis_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] 已达到最大舌象采集数量({self.camera_thread.max_tongue_crops}张)")
        self.status_bar.showMessage("舌象采集已完成")



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

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置现代风格
    app.setStyle("Fusion")
    
    # 创建主窗口实例
    main_window = MainUI()  # 直接使用MainUI作为主窗口
    
    # 显示窗口
    main_window.show()
    
    # 启动事件循环
    sys.exit(app.exec_())
