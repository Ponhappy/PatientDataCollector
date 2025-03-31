from ultralytics import YOLO
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QComboBox,
    QLabel, QTextBrowser, QInputDialog, QMessageBox, QDialog, QLineEdit,
    QFormLayout, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QWidget,
    QStatusBar, QToolButton, QGroupBox, QTabWidget, QScrollArea, QGridLayout,
    QStyleFactory, QSplitter, QTextEdit, QRadioButton
)
import sys
from PyQt5.QtWidgets import QWidget
import os
import threading
import serial
import serial.tools.list_ports
import json
from datetime import datetime
from finger_thread import FingerDataThread
from camera_thread import CameraThread
from chat_thread import ChatThread
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QImage
from tongue_diagnose_model.tongue_diagnose import tongue_diagnose_sum
from face_diagnose_model.face_diagnose import face_diagnose_sum
from chat_model.cloud_chat import CloudChat
from chat_model.local_chat import LocalChat
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
        self.setWindowTitle("智能中医四诊系统")
        self.setMinimumSize(1200, 800)  # 设置最小窗口大小
        
        # 初始化患者数据存储路径
        self.patient_list_dp = os.path.join(os.path.expanduser("~"), "patient_data")  # 默认存储在用户目录下
        if not os.path.exists(self.patient_list_dp):
            os.makedirs(self.patient_list_dp)  # 自动创建目录
        
        # 添加舌诊处理标志
        self.tongue_diagnosed = False
        self.face_diagnosed = False
        self.diagnosis_in_progress = False
        self.use_original_frame = True  # 设置为True使用原始帧，False使用裁剪图像
        
        # 创建并设置中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)
        
        # 创建左侧控制面板和中间显示区域以及右侧诊断报告区域
        self.setup_control_panel()
        self.setup_main_area()
        
        # 创建底部状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪")
        
        # 加载并应用样式表
        self.apply_stylesheet()
        
        # 初始化变量
        self.camera_thread = None
        self.finger_thread = None
        self.camera_index = 0
        # 初始化串口属性
        self.finger_serial_port = None
        
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
        
        # 移除腕带串口，保留指夹串口
        self.finger_serial_label = QtWidgets.QLabel("选择指夹串口:")
        self.finger_serial_combo = QtWidgets.QComboBox()
        self.finger_serial_combo.setObjectName("deviceCombo")
        self.confirm_finger_btn = QtWidgets.QPushButton("确认")
        self.confirm_finger_btn.clicked.connect(self.confirm_finger_selection)
        device_layout.addWidget(self.finger_serial_label, 1, 0)
        device_layout.addWidget(self.finger_serial_combo, 1, 1)
        device_layout.addWidget(self.confirm_finger_btn, 1, 2)
        
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
        
        # 添加诊断控制组
        self.diagnosis_group = QGroupBox("诊断控制")
        self.diagnosis_layout = QVBoxLayout()
        
        # 创建舌诊和面诊按钮
        self.tongue_diagnosis_btn = QPushButton("开始舌诊")
        self.tongue_diagnosis_btn.clicked.connect(self.start_tongue_diagnosis)
        
        self.face_diagnosis_btn = QPushButton("开始面诊")
        self.face_diagnosis_btn.clicked.connect(self.start_face_diagnosis)
        
        self.pause_camera_btn = QPushButton("暂停摄像头")
        self.pause_camera_btn.clicked.connect(self.toggle_camera_pause)
        self.pause_camera_btn.setEnabled(False)  # 初始状态禁用
        
        # 添加导出报告按钮
        self.export_report_btn = QPushButton("导出诊断报告")
        self.export_report_btn.clicked.connect(self.export_html_report)
        
        # 添加按钮到布局
        self.diagnosis_layout.addWidget(self.tongue_diagnosis_btn)
        self.diagnosis_layout.addWidget(self.face_diagnosis_btn)
        self.diagnosis_layout.addWidget(self.pause_camera_btn)
        self.diagnosis_layout.addWidget(self.export_report_btn)
        
        self.diagnosis_group.setLayout(self.diagnosis_layout)
        operation_layout.addWidget(self.diagnosis_group)
        
        # 将所有组添加到控制面板
        control_layout.addWidget(user_group)
        control_layout.addWidget(device_group)
        control_layout.addWidget(operation_group)
        control_layout.addStretch()
        
        # 将控制面板添加到主布局
        self.main_layout.addWidget(self.control_panel)

    def setup_main_area(self):
        """设置中央和右侧显示区域"""
        # 创建一个水平分割器，用于中央显示区和右侧诊断报告区
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # 创建中央显示区域
        self.display_area = QWidget()
        self.display_area.setObjectName("displayArea")
        display_layout = QVBoxLayout(self.display_area)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(10)
        
        # 创建垂直分割器，用于上部相机视图和下部聊天窗口
        self.display_splitter = QSplitter(Qt.Vertical)
        
        # 创建相机视图区域
        camera_view_container = QWidget()
        camera_view_layout = QVBoxLayout(camera_view_container)
        camera_view_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建画面显示区域
        self.video_display = QLabel("等待摄像头启动...")
        self.video_display.setObjectName("videoDisplay")
        self.video_display.setAlignment(Qt.AlignCenter)
        self.video_display.setMinimumSize(640, 480)
        self.video_display.setStyleSheet("background-color: #2c3e50; color: white;")
        
        camera_view_layout.addWidget(self.video_display)
        
        # 创建聊天窗口区域
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(5, 5, 5, 5)
        
        # 添加模型选择按钮组
        model_selection_widget = QWidget()
        model_selection_layout = QHBoxLayout(model_selection_widget)
        model_selection_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cloud_model_radio = QRadioButton("云端模型")
        self.local_model_radio = QRadioButton("本地模型")
        self.cloud_model_radio.setChecked(True)  # 默认选择云端模型
        
        # 添加API设置按钮
        self.api_settings_btn = QPushButton("API设置")
        self.api_settings_btn.clicked.connect(self.show_api_settings)
        
        model_selection_layout.addWidget(self.cloud_model_radio)
        model_selection_layout.addWidget(self.local_model_radio)
        model_selection_layout.addWidget(self.api_settings_btn)
        model_selection_layout.addStretch()
        
        # 聊天历史显示
        self.chat_history = QTextBrowser()
        self.chat_history.setObjectName("chatHistory")
        self.chat_history.setMinimumHeight(150)
        
        # 聊天输入区域
        chat_input_container = QWidget()
        chat_input_layout = QHBoxLayout(chat_input_container)
        chat_input_layout.setContentsMargins(0, 0, 0, 0)
        
        self.chat_input = QTextEdit()
        self.chat_input.setObjectName("chatInput")
        self.chat_input.setMaximumHeight(80)
        self.chat_input.setPlaceholderText("在此输入问题...")
        
        self.chat_send_btn = QPushButton("发送")
        self.chat_send_btn.setObjectName("chatSendBtn")
        self.chat_send_btn.clicked.connect(self.send_chat_message)
        
        chat_input_layout.addWidget(self.chat_input, 4)
        chat_input_layout.addWidget(self.chat_send_btn, 1)
        
        # 添加到聊天布局
        chat_layout.addWidget(model_selection_widget)
        chat_layout.addWidget(self.chat_history, 3)
        chat_layout.addWidget(chat_input_container, 1)
        
        # 添加到显示分割器
        self.display_splitter.addWidget(camera_view_container)
        self.display_splitter.addWidget(chat_container)
        self.display_splitter.setStretchFactor(0, 3)
        self.display_splitter.setStretchFactor(1, 1)
        
        # 添加到显示区域布局
        display_layout.addWidget(self.display_splitter)
        
        # 创建右侧诊断报告区域
        self.diagnosis_report_area = QWidget()
        self.diagnosis_report_area.setObjectName("diagnosisReportArea")
        self.diagnosis_report_area.setMinimumWidth(400)
        
        diagnosis_report_layout = QVBoxLayout(self.diagnosis_report_area)
        diagnosis_report_layout.setContentsMargins(0, 0, 0, 0)
        
        # 诊断报告标题
        report_title = QLabel("诊断报告")
        report_title.setObjectName("reportTitle")
        report_title.setAlignment(Qt.AlignCenter)
        
        # 诊断报告内容 - 使用QTextBrowser以支持HTML内容和滚动
        self.diagnosis_report = QTextBrowser()
        self.diagnosis_report.setObjectName("diagnosisReport")
        self.diagnosis_report.setOpenExternalLinks(True)  # 允许打开外部链接
        
        diagnosis_report_layout.addWidget(report_title)
        diagnosis_report_layout.addWidget(self.diagnosis_report)
        
        # 添加到主分割器
        self.main_splitter.addWidget(self.display_area)
        self.main_splitter.addWidget(self.diagnosis_report_area)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)
        
        # 将主分割器添加到主布局
        self.main_layout.addWidget(self.main_splitter)

        # 初始化聊天模型
        self.init_chat_models()

    def init_chat_models(self):
        """初始化聊天模型"""
        # 读取配置或使用默认值
        config_path = os.path.join(os.path.expanduser("~"), ".tcm_config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = {
                    "api_key": "sk-d69f89a753d74b399a9404194d611aaa",  # 默认API密钥
                    "base_url": "https://api.deepseek.com",            # 默认API基址
                    "prompt": "你是一个中医专家，熟悉舌诊和脉诊等中医诊断方法。请基于医学知识，对患者的问题给出专业的意见。",
                    "model": "deepseek-chat"
                }
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"加载配置文件出错: {e}")
            config = {
                "api_key": "sk-d69f89a753d74b399a9404194d611aaa",
                "base_url": "https://api.deepseek.com",
                "prompt": "你是一个中医专家，熟悉舌诊和脉诊等中医诊断方法。请基于医学知识，对患者的问题给出专业的意见。",
                "model": "deepseek-chat"
            }
        
        self.chat_config = config
        self.cloud_chat = None  # 延迟初始化，等到实际需要时创建
        self.local_chat = None
        self.current_chat_history_file = None

    def show_api_settings(self):
        """显示API设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("API设置")
        layout = QFormLayout(dialog)
        
        # API密钥输入
        api_key_input = QLineEdit(self.chat_config.get("api_key", ""))
        layout.addRow("API密钥:", api_key_input)
        
        # 基础URL输入
        base_url_input = QLineEdit(self.chat_config.get("base_url", "https://api.deepseek.com"))
        layout.addRow("基础URL:", base_url_input)
        
        # 模型提示词
        prompt_input = QTextEdit()
        prompt_input.setText(self.chat_config.get("prompt", "你是一个中医专家"))
        prompt_input.setMaximumHeight(100)
        layout.addRow("系统提示词:", prompt_input)
        
        # 模型选择
        model_input = QComboBox()
        model_input.addItems(["deepseek-chat", "deepseek-reasoner"])
        model_input.setCurrentText(self.chat_config.get("model", "deepseek-chat"))
        layout.addRow("模型:", model_input)
        
        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            # 保存设置
            self.chat_config["api_key"] = api_key_input.text()
            self.chat_config["base_url"] = base_url_input.text()
            self.chat_config["prompt"] = prompt_input.toPlainText()
            self.chat_config["model"] = model_input.currentText()
            
            # 保存到配置文件
            config_path = os.path.join(os.path.expanduser("~"), ".tcm_config.json")
            try:
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.chat_config, f, ensure_ascii=False, indent=2)
                
                # 重置模型实例，使其使用新设置
                self.cloud_chat = None
                QMessageBox.information(self, "设置保存", "API设置已保存成功!")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存设置时出错:\n{str(e)}")
    
    def get_or_create_chat_model(self):
        """获取或创建当前选择的聊天模型实例"""
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QMessageBox.warning(self, '无用户', '请先添加用户再开始聊天。')
            return None
            
        # 设置当前用户的历史记录文件
        user_dir = os.path.join(self.patient_list_dp, selected_user)
        os.makedirs(user_dir, exist_ok=True)
        
        if self.cloud_model_radio.isChecked():
            # 使用云端模型
            history_file = os.path.join(user_dir, "cloud_chat_history.json")
            if self.cloud_chat is None or self.current_chat_history_file != history_file:
                self.cloud_chat = CloudChat(
                    api_key=self.chat_config["api_key"],
                    base_url=self.chat_config["base_url"],
                    prompt=self.chat_config["prompt"],
                    model=self.chat_config["model"],
                    history_file=history_file
                )
                self.current_chat_history_file = history_file
            return self.cloud_chat
        else:
            # 使用本地模型
            history_file = os.path.join(user_dir, "local_chat_history.json")
            if self.local_chat is None or self.current_chat_history_file != history_file:
                self.local_chat = LocalChat()  # 本地模型可能需要其他参数
                self.current_chat_history_file = history_file
            return self.local_chat
    
    def send_chat_message(self):
        """发送聊天消息并获取回答"""
        user_message = self.chat_input.toPlainText().strip()
        if not user_message:
            return
            
        # 显示用户消息
        self.chat_history.append(f'<p style="text-align: right;"><b>您:</b> {user_message}</p>')
        self.chat_input.clear()
        
        # 获取当前选择的聊天模型
        chat_model = self.get_or_create_chat_model()
        if chat_model is None:
            return
            
        # 设置"等待中"状态
        self.chat_send_btn.setEnabled(False)
        self.chat_history.append('<p><b>AI助手:</b> <i>思考中...</i></p>')
        
        # 使用线程处理聊天请求以避免UI冻结
        self.chat_thread = ChatThread(chat_model, user_message)
        self.chat_thread.response_ready.connect(self.display_chat_response)
        self.chat_thread.start()
    
    def display_chat_response(self, response):
        """显示AI回复"""
        # 移除"思考中"文本
        current_html = self.chat_history.toHtml()
        current_html = current_html.replace('<p><b>AI助手:</b> <i>思考中...</i></p>', '')
        self.chat_history.setHtml(current_html)
        
        # 显示实际回复
        self.chat_history.append(f'<p><b>AI助手:</b> {response}</p>')
        self.chat_send_btn.setEnabled(True)
        
        # 滚动到底部
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum()
        )

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
        # 只填充指夹串口
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

    # 只启动指夹传感器，移除腕带相关代码
    def start_sensors_only(self):
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QtWidgets.QMessageBox.warning(self, '无用户', '请先添加用户。')
            return
        user_dir = os.path.join(self.patient_list_dp, selected_user)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        self.patient_id = selected_user  # 使用用户名作为 patient_id

        # 指夹传感器
        if self.finger_serial_port and self.finger_serial_port != "没有可用的串口":
            if self.finger_thread is None:
                self.finger_thread = FingerDataThread(
                    serial_port=self.finger_serial_port,
                    baudrate=115200,
                )
                self.finger_thread.csv_filename = os.path.join(user_dir, "finger_pulse.csv")
                self.finger_thread.data_received.connect(self.show_sensors_report)
                self.finger_thread.start()
                print("指夹数据采集线程已启动。")
            else:
                print("指夹数据采集线程已在运行。")
        else:
            print("指夹串口未选择或初始化失败。")

    def show_sensors_report(self, report):
        # 将报告内容直接添加到诊断报告中
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_report = f"<p><b>[{timestamp}]</b> {report}</p>"
        
        # 在诊断报告中添加
        self.diagnosis_report.append(formatted_report)
        
        # 更新状态栏信息
        self.status_bar.showMessage("智能脉诊已完成")

    def stop_sensors(self):
        if self.finger_thread is not None:
            self.finger_thread.stop()
            self.finger_thread = None
            print("指夹数据采集线程已停止。")

    def start_tongue_diagnosis(self):
        """开始舌诊分析流程"""
        # 确保摄像头是运行状态
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.start_camera_only()  # 启动摄像头
        
        # 重置诊断状态
        self.tongue_diagnosed = False
        self.diagnosis_in_progress = True
        self.pause_camera_btn.setEnabled(True)
        
        # 确保舌头检测已开启
        if self.camera_thread:
            self.camera_thread.set_tongue_detection_enabled(True)
            self.camera_thread.first_image_sent = False  # 重置图像发送标志
        
        self.diagnosis_report.append(f"[{datetime.now().strftime('%H:%M:%S')}] 开始舌诊分析...")
        self.status_bar.showMessage("请伸出舌头，系统正在进行舌诊分析...")

    def start_face_diagnosis(self):
        """开始面诊分析流程"""
        # 确保摄像头是运行状态
        if self.camera_thread is None or not self.camera_thread.isRunning():
            self.start_camera_only()  # 启动摄像头
        
        # 重置面诊状态
        self.face_diagnosed = False
        self.diagnosis_in_progress = True
        self.pause_camera_btn.setEnabled(True)
        
        # 启用面诊检测
        if self.camera_thread:
            self.camera_thread.set_face_detection_enabled(True)
            self.camera_thread.set_diagnosis_completed(False)  # 重置诊断完成标志
        
        # 创建用户面诊文件夹
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QtWidgets.QMessageBox.warning(self, '无用户', '请先添加用户。')
            return
        
        user_dir = os.path.join(self.patient_list_dp, selected_user)
        face_dir = os.path.join(user_dir, "face_images")
        if not os.path.exists(face_dir):
            os.makedirs(face_dir)
        
        # 确保faceseg目录存在
        faceseg_dir = os.path.join(user_dir, "faceseg")
        if not os.path.exists(faceseg_dir):
            os.makedirs(faceseg_dir)
        faceseg_roi_dir = os.path.join(faceseg_dir, "roi_images")
        if not os.path.exists(faceseg_roi_dir):
            os.makedirs(faceseg_roi_dir)
        
        self.diagnosis_report.append(f"[{datetime.now().strftime('%H:%M:%S')}] 开始面诊分析...")
        self.status_bar.showMessage("请面向摄像头，系统正在进行面诊分析...")

    def toggle_camera_pause(self):
        """暂停/恢复摄像头线程"""
        if not self.camera_thread:
            return
        
        if self.camera_thread.isRunning():
            # 暂停摄像头线程
            self.camera_thread.pause()
            self.pause_camera_btn.setText("恢复摄像头")
            self.status_bar.showMessage("摄像头已暂停")
        else:
            # 恢复摄像头线程
            self.camera_thread.resume()
            self.pause_camera_btn.setText("暂停摄像头")
            self.status_bar.showMessage("摄像头已恢复")

    def perform_tongue_diagnosis(self, image_path, is_original_frame=True):
        """执行舌诊分析
        
        Args:
            image_path: 图像文件路径
            is_original_frame: 是否是原始帧图像(True)还是裁剪图像(False)
        """
        # 调用舌头诊断函数
        color_report, coating_report, cancer_report, tongue_annotated, diagnosis, treatment = tongue_diagnose_sum(image_path, is_original_frame)
        
        # 创建HTML诊断报告
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        html_report = f"""
        <h3>舌诊分析报告 - {timestamp}</h3>
        <div style="display: flex; margin-bottom: 15px;">
            <img src="file:///{tongue_annotated}" style="max-width: 300px; margin-right: 15px;">
            <div>
                <p><b>诊断结论:</b> {diagnosis}</p>
                <p><b>治疗建议:</b> {treatment}</p>
            </div>
        </div>
        <h4>详细分析</h4>
        <p><b>舌色分析:</b> {color_report}</p>
        <p><b>舌苔分析:</b> {coating_report}</p>
        <p><b>异常检测:</b> {cancer_report}</p>
        <hr>
        """
        
        # 添加到诊断报告
        self.diagnosis_report.append(html_report)
        
        # 标记已完成诊断
        self.tongue_diagnosed = True
        self.diagnosis_in_progress = False
        
        # 通知摄像头线程诊断已完成
        if self.camera_thread:
            self.camera_thread.set_diagnosis_completed(True)
        
        self.status_bar.showMessage("舌诊分析已完成")

    def perform_face_diagnosis(self, image_path):
        """执行面诊分析"""
        # 调用面诊分析函数
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QtWidgets.QMessageBox.warning(self, '无用户', '请先添加用户。')
            return
        
        user_dir = os.path.join(self.patient_list_dp, selected_user)
        diagnosis_report, annotated_img_path = face_diagnose_sum(image_path, user_dir)
        
        # 创建HTML诊断报告
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        html_report = f"""
        <h3>面诊分析报告 - {timestamp}</h3>
        <div style="display: flex; margin-bottom: 15px;">
            <img src="file:///{annotated_img_path}" style="max-width: 300px; margin-right: 15px;">
            <div>
                <p>{diagnosis_report}</p>
            </div>
        </div>
        <hr>
        """
        
        # 添加到诊断报告
        self.diagnosis_report.append(html_report)
        
        # 标记已完成诊断
        self.face_diagnosed = True
        self.diagnosis_in_progress = False
        
        # 通知摄像头线程诊断已完成
        if self.camera_thread:
            self.camera_thread.face_diagnosed = True
        
        self.status_bar.showMessage("面诊分析已完成")

    def closeEvent(self, event):
        # 当窗口关闭时，确保所有线程都被正确停止
        if self.finger_thread is not None:
            self.finger_thread.stop()
        if self.camera_thread is not None:
            self.camera_thread.stop()
        event.accept()

    # 新增：导出HTML报告
    def export_html_report(self):
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QMessageBox.warning(self, '无用户', '请先添加用户。')
            return
            
        # 获取HTML内容
        html_content = self.diagnosis_report.toHtml()
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存诊断报告", 
            os.path.join(self.patient_list_dp, selected_user, f"{selected_user}_诊断报告.html"),
            "HTML Files (*.html)")
            
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(html_content)
                QMessageBox.information(self, "导出成功", f"诊断报告已成功导出到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"保存报告时出错:\n{str(e)}")

    # 仅启动摄像头
    def start_camera_only(self):
        # 重置舌诊处理标志
        self.tongue_diagnosed = False
        
        # 确保摄像头索引有效
        if self.camera_combo.currentText() == "未检测到摄像头":
            self.status_bar.showMessage("请先选择一个摄像头")
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
        user_dir = os.path.join(self.patient_list_dp, selected_user)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        self.patient_id = selected_user  # 使用用户名作为 patient_id

        # 启动摄像头线程
        if self.camera_thread is None:
            self.camera_thread = CameraThread(
                save_dir=user_dir,
                crop_tongue_interval=5,
                camera_index=self.camera_index
            )
            
            # 连接信号
            self.camera_thread.frame_received.connect(self.display_camera_frame)
            self.camera_thread.guidance_message.connect(self.show_guidance)
            self.camera_thread.crop_tongue_saved_path.connect(self.handle_new_crop_image)
            self.camera_thread.original_frame_saved_path.connect(self.handle_original_frame)
            self.camera_thread.max_images_reached.connect(self.handle_max_images_reached)
            self.camera_thread.original_frame_saved_path.connect(self.handle_face_image)
            
            # 配置摄像头线程 
            self.camera_thread.set_frames_to_skip(15)
            self.camera_thread.set_crop_tongue_interval(3)
            
            # 连接诊断状态更新
            self.camera_thread.set_tongue_detection_enabled(True)
            self.camera_thread.set_diagnosis_completed(False)  # 重置诊断完成状态
            
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

    def show_guidance(self, message):
        """显示检测引导提示"""
        print("调用show_guidance函数")
        
            
        # 更新状态栏
        status_msg = message
        self.status_bar.showMessage(status_msg)
            
            

    def handle_new_crop_image(self, crop_path):
        """处理新裁剪的舌头图像"""
        # 如果已经进行过诊断或诊断未开始，则忽略后续的图像
        if self.tongue_diagnosed or not self.diagnosis_in_progress:
            return
        
        if not os.path.exists(crop_path):
            print(f"文件不存在: {crop_path}")
            return
        
        # 如果配置使用原始帧，则跳过裁剪图像处理
        if self.use_original_frame:
            return
        
        # 显示第一张裁剪图
        if not hasattr(self, 'latest_crop_path'):
            self.display_first_crop(crop_path)
        
        # 执行舌诊分析
        self.perform_tongue_diagnosis(crop_path)
        
        # 标记已完成诊断
        self.tongue_diagnosed = True
        self.diagnosis_in_progress = False
        
        # 通知摄像头线程诊断已完成
        if self.camera_thread:
            self.camera_thread.set_diagnosis_completed(True)
        
        self.status_bar.showMessage("舌诊分析已完成")
        print("使用裁剪图像完成舌诊分析")
        
    def handle_original_frame(self, original_path, crop_path):
        """处理保存的原始帧图像"""
        # 如果已经进行过诊断或诊断未开始，则忽略
        if self.tongue_diagnosed or not self.diagnosis_in_progress:
            return
        
        if not os.path.exists(original_path):
            print(f"原始帧文件不存在: {original_path}")
            return
        
        # 如果配置不使用原始帧进行舌诊，则跳过处理
        if not self.use_original_frame:
            return
        
        # 显示裁剪图
        if not hasattr(self, 'latest_crop_path'):
            self.display_first_crop(crop_path)
        
        # 使用原始帧进行舌诊分析
        self.perform_tongue_diagnosis(original_path)
        
        # 标记已完成诊断
        self.tongue_diagnosed = True
        self.diagnosis_in_progress = False
        
        # 通知摄像头线程诊断已完成
        if self.camera_thread:
            self.camera_thread.set_diagnosis_completed(True)
        
        self.status_bar.showMessage("舌诊分析已完成")
        print("使用原始帧完成舌诊分析")
        
    def display_first_crop(self, path):
        """显示第一张裁剪图像"""
        pixmap = QPixmap(path)
        self.video_display.setPixmap(pixmap.scaled(self.video_display.width(), self.video_display.height(), 
                                   QtCore.Qt.KeepAspectRatio))
        self.latest_crop_path = path
        
    def handle_max_images_reached(self):
        """处理达到最大图像数量的情况"""
        self.diagnosis_report.append(f"[{datetime.now().strftime('%H:%M:%S')}] 已达到最大舌象采集数量({self.camera_thread.max_tongue_crops}张)")
        self.status_bar.showMessage("舌象采集已完成")
        
        # 自动暂停摄像头
        if self.camera_thread and self.camera_thread.isRunning():
            self.toggle_camera_pause()
            self.diagnosis_report.append(f"[{datetime.now().strftime('%H:%M:%S')}] 诊断完成，摄像头已自动暂停")

    def handle_face_image(self, image_path):
        """处理面诊图像"""
        # 如果已经进行过诊断或诊断未开始，则忽略
        if self.face_diagnosed or not self.diagnosis_in_progress:
            return
        
        if not os.path.exists(image_path):
            print(f"面诊图像不存在: {image_path}")
            return
        
        # 执行面诊分析
        self.perform_face_diagnosis(image_path)
        
        # 标记已完成诊断
        self.face_diagnosed = True
        self.diagnosis_in_progress = False
        
        # 通知摄像头线程诊断已完成
        if self.camera_thread:
            self.camera_thread.face_diagnosed = True
        
        self.status_bar.showMessage("面诊分析已完成")


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
