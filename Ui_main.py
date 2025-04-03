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
from datetime import datetime, timedelta
from finger_thread import FingerDataThread
from camera_thread import CameraThread
from chat_thread import ChatThread
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap, QImage
from tongue_diagnose_model.tongue_diagnose import tongue_diagnose_sum
from face_diagnose_model.face_diagnose import face_diagnose_sum
from chat_model.cloud_chat import CloudChat
from chat_model.local_chat import LocalChat
import pdfkit
# import resources_rc  # 资源文件

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
import traceback
import pyqtgraph as pg
import numpy as np

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
        self.showMaximized()   # 设置窗口最大化
        
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

        self.latest_crop_tongue_path = None  # 存储最新的舌头裁剪图像路径

        # 将init_chat_models替换为init_chat_config
        self.init_chat_config()

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
        
        # 指夹串口
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
        
        self.pause_camera_btn = self.create_action_button("暂停摄像头", ":/icons/stop_camera.png")
        self.pause_camera_btn.clicked.connect(self.toggle_camera_pause)
        self.pause_camera_btn.setEnabled(False)  # 初始状态禁用
        
        self.start_sensors_btn = self.create_action_button("开始脉象采集", ":/icons/sensor.png")
        self.start_sensors_btn.clicked.connect(self.start_sensors_only)
        
        self.stop_sensors_btn = self.create_action_button("停止脉象采集", ":/icons/stop.png")
        self.stop_sensors_btn.clicked.connect(self.stop_sensors)
        
        self.refresh_devices_btn = self.create_action_button("刷新设备列表", ":/icons/refresh.png")
        self.refresh_devices_btn.clicked.connect(self.refresh_devices)
        

        
        # 添加按钮到操作布局
        operation_layout.addWidget(self.start_camera_btn)
        operation_layout.addWidget(self.pause_camera_btn)
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
        
        # self.sum_diagnosis_btn = QPushButton("AI综合诊断")
        # self.sum_diagnosis_btn.clicked.connect(self.start_ai_diagnosis)
        
        # self.pause_camera_btn = QPushButton("暂停摄像头")
        # self.pause_camera_btn.clicked.connect(self.toggle_camera_pause)
        # self.pause_camera_btn.setEnabled(False)  # 初始状态禁用
        
        # 添加导出报告按钮
        self.export_report_btn = QPushButton("导出诊断报告")
        self.export_report_btn.clicked.connect(self.export_report)
        
        # 添加按钮到布局
        self.diagnosis_layout.addWidget(self.tongue_diagnosis_btn)
        self.diagnosis_layout.addWidget(self.face_diagnosis_btn)
        # self.diagnosis_layout.addWidget(self.sum_diagnosis_btn)
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
        
        # 创建实时脉搏波形显示区域
        finger_pulse_container = QWidget()
        finger_pulse_layout = QVBoxLayout(finger_pulse_container)
        finger_pulse_layout.setContentsMargins(0, 0, 0, 0)

        # 创建绘图窗口
        self.graphWidget = pg.PlotWidget()
        # self.setCentralWidget(self.graphWidget)
        self.graphWidget.setBackground("w")  # 设置白色背景
        self.graphWidget.setTitle("串口数据实时波形", color="black", size="15pt")
        self.graphWidget.setLabel("left", "信号值", color="black", size="12pt")
        self.graphWidget.setLabel("bottom", "时间 (s)", color="black", size="12pt")
        self.graphWidget.showGrid(x=True, y=True)
        # 获取X轴和Y轴的AxisItem对象
        x_axis = self.graphWidget.getAxis('bottom')
        y_axis = self.graphWidget.getAxis('left')

        # 隐藏刻度标签，同时保留刻度
        x_axis.setTicks([[]])
        y_axis.setTicks([[]])
        # 固定 X 轴和 Y 轴范围
        self.cnt=0
        self.x_time = []
        self.y_wave = []
        self.time_window=10
        self.graphWidget.setXRange(0, self.time_window)  # 横轴固定 -10 到 0 秒
        self.graphWidget.setYRange(0, 150)   # 纵轴范围根据实际数据调整
        self.finger_start_time=None
        finger_pulse_layout.addWidget(self.graphWidget)



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
        self.chat_input.setPlaceholderText("请在此描述您的症状...")
        
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
        self.display_splitter.addWidget(finger_pulse_container)
        self.display_splitter.addWidget(chat_container)
        
        self.display_splitter.setStretchFactor(0, 3)
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
        report_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        
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
        self.init_chat_config()

    def init_chat_config(self):
        """初始化聊天配置，但不创建模型实例"""
        # 读取配置或使用默认值
        config_path = os.path.join(os.path.expanduser("~"), ".tcm_config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.chat_config = json.load(f)
            else:
                # 默认配置
                self.chat_config = {
                    "api_key": "sk-d69f89a753d74b399a9404194d611aaa",
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-reasoner",
                    "system_prompt": """你是一位经验丰富的中医AI助手，帮助医生和患者理解中医诊断结果和健康建议。

你有以下特点和能力：
1. 可以根据舌诊、面诊和脉诊的机器检测结果提供专业分析
2. 会参考患者的历史对话和症状描述
3. 提供符合中医理论的辨证分析和调理建议
4. 回答既专业又通俗易懂，能够解释专业术语
5. 在合适的情况下建议就医，不会替代正规医疗诊断
6. 对于患者的痛苦表示理解和同情

当你看到[舌诊报告]、[面诊报告]或[脉诊报告]标记时，这表示这些是由专业中医诊断设备测量的结果，具有较高的参考价值。"""
                }
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.chat_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"加载配置文件出错: {e}")
            # 出错时使用默认配置
            self.chat_config = {
                "api_key": "sk-d69f89a753d74b399a9404194d611aaa",
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-reasoner",
                "system_prompt": "你是一位专业的中医AI助手..."
            }
        
        # 初始化当前历史文件路径为None，表示尚未为特定用户创建聊天模型
        self.current_chat_history_file = None
        self.cloud_chat = None
        self.local_chat = None

    def show_api_settings(self):
        """显示API设置对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("API设置")
        dialog.setMinimumWidth(400)
        
        layout = QFormLayout(dialog)
        
        # API密钥
        api_key_input = QLineEdit()
        api_key_input.setText(self.chat_config.get("api_key", ""))
        api_key_input.setEchoMode(QLineEdit.Password)
        layout.addRow("API密钥:", api_key_input)
        
        # 基础URL
        base_url_input = QLineEdit()
        base_url_input.setText(self.chat_config.get("base_url", "https://api.deepseek.com/v1"))
        layout.addRow("基础URL:", base_url_input)
        
        # 模型选择
        model_input = QComboBox()
        model_input.addItems(["deepseek-reasoner", "deepseek-chat"])
        model_input.setCurrentText(self.chat_config.get("model", "deepseek-reasoner"))
        layout.addRow("模型:", model_input)
        
        # 系统提示词
        system_prompt_input = QTextEdit()
        system_prompt_input.setText(self.chat_config.get("system_prompt", ""))
        system_prompt_input.setMinimumHeight(150)
        layout.addRow("系统提示词:", system_prompt_input)
        
        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        # 显示对话框
        result = dialog.exec_()
        if result == QDialog.Accepted:
            # 更新配置
            self.chat_config["api_key"] = api_key_input.text()
            self.chat_config["base_url"] = base_url_input.text()
            self.chat_config["model"] = model_input.currentText()
            self.chat_config["system_prompt"] = system_prompt_input.toPlainText()
            
            # 保存配置
            config_path = os.path.join(os.path.expanduser("~"), ".tcm_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.chat_config, f, ensure_ascii=False, indent=2)
            
            # 重置聊天模型实例，以便下次使用时使用新配置
            self.cloud_chat = None
            self.local_chat = None
            self.current_chat_history_file = None
            
            QMessageBox.information(self, "成功", "API设置已更新")

    def get_or_create_chat_model(self):
        """获取或创建聊天模型，确保使用正确的用户历史记录"""
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QMessageBox.warning(self, '无用户', '请先添加用户。')
            return None
        
        # 为当前用户创建历史记录目录
        user_dir = os.path.join(self.patient_list_dp, selected_user)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        # 用户特定的历史记录文件
        history_file = os.path.join(user_dir, "chat_history.json")
        
        # 根据用户选择初始化相应的聊天模型
        if self.cloud_model_radio.isChecked():
            # 检查是否有现有的模型实例以及是否需要创建新的
            if not self.cloud_chat or self.current_chat_history_file != history_file:
                # 从配置获取参数
                api_key = self.chat_config.get('api_key', '')
                base_url = self.chat_config.get('base_url', 'https://api.deepseek.com/v1')
                model = self.chat_config.get('model', 'deepseek-reasoner')
                system_prompt = self.chat_config.get('system_prompt', '')
                
                # 初始化云端聊天模型
                from chat_model.cloud_chat import CloudChat
                self.cloud_chat = CloudChat(
                    api_key=api_key, 
                    base_url=base_url,
                    model=model,
                    system_prompt=system_prompt,
                    history_file=history_file
                )
                self.current_chat_history_file = history_file
            return self.cloud_chat
        else:
            # 本地模型初始化
            if not self.local_chat or self.current_chat_history_file != history_file:
                system_prompt = self.chat_config.get('system_prompt', '')
                from chat_model.local_chat import LocalChat
                self.local_chat = LocalChat(
                    system_prompt=system_prompt,
                    history_file=history_file
                )
                self.current_chat_history_file = history_file
            return self.local_chat

    def send_chat_message(self):
        """发送聊天消息"""
        message = self.chat_input.toPlainText().strip()
        if not message:
            return
        
        # 获取或创建聊天模型
        chat_model = self.get_or_create_chat_model()
        if not chat_model:
            return
        
        # 显示用户消息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chat_history.append(f'<p style="text-align: right;"><span style="color: #888; font-size: 0.8em;">{timestamp}</span><br><b>您:</b> {message}</p>')
        
        # 清空输入框
        self.chat_input.clear()
        
        # 显示"思考中"
        self.chat_history.append('<p><b>AI助手:</b> <i>思考中...</i></p>')
        self.chat_send_btn.setEnabled(False)
        
        # 创建线程处理请求
        self.chat_thread = ChatThread(chat_model, message, timestamp)
        self.chat_thread.response_ready.connect(self.handle_chat_response)
        self.chat_thread.start()

    def handle_chat_response(self, response_data):
        """显示AI回复"""
        response, timestamp = response_data
        
        # 移除"思考中"文本
        current_html = self.chat_history.toHtml()
        current_html = current_html.replace('<p><b>AI助手:</b> <i>思考中...</i></p>', '')
        self.chat_history.setHtml(current_html)
        
        # 解析不同模型的响应
        if "|||" in response:  # ds_r1格式
            think, answer = response.split("|||", 1)
            formatted = f'''
            <p><span style="color: #888; font-size: 0.8em;">{timestamp}</span></p>
            <p style="color: #666; font-size: 0.9em; margin-bottom: 5px;">{think}</p>
            <div style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
                <b>AI助手:</b> 
                <div style="font-family: 'Consolas', monospace; white-space: pre-wrap;">{answer}</div>
            </div>
            '''
        else:  # ds_v3格式
            formatted = f'''
            <p><span style="color: #888; font-size: 0.8em;">{timestamp}</span></p>
            <div style="background: #f5f5f5; padding: 10px; border-radius: 5px;">
                <b>AI助手:</b> 
                <div style="font-family: 'Consolas', monospace; white-space: pre-wrap;">{response}</div>
            </div>
            '''
        
        self.chat_history.append(formatted)
        self.chat_send_btn.setEnabled(True)
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
        """填充可用的串口列表并自动选择第一个"""
        self.finger_serial_combo.clear()
        ports = serial.tools.list_ports.comports()
        
        # 过滤有效串口（可根据实际设备VID/PID调整）
        valid_ports = []
        for port in ports:
            # 示例：如果知道设备VID/PID可在此过滤
            # if port.vid == 0x1234 and port.pid == 0x5678:
            valid_ports.append(port.device)
        
        if valid_ports:
            self.finger_serial_combo.addItems(valid_ports)
            # 自动选择第一个有效设备
            self.finger_serial_combo.setCurrentIndex(0)
            self.finger_serial_port = valid_ports[0]
            print(f"自动选择串口: {self.finger_serial_port}")
            self.status_bar.showMessage(f"已自动选择串口 {self.finger_serial_port}", 3000)
        else:
            self.finger_serial_combo.addItem("没有可用的串口")
            self.finger_serial_port = None
            self.status_bar.showMessage("未检测到有效串口设备", 5000)
        

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

    # 只启动指夹传感器
    def start_sensors_only(self):
        self.finger_start_time = datetime.now()
        # 添加时间戳记录
        self.diagnosis_report.append(f"[{datetime.now().strftime('%H:%M:%S')}] 开始脉诊分析...")
        self.status_bar.showMessage("请将手指放在传感器上，系统正在进行脉诊分析...")
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
                    baudrate=38400,
                )
                self.finger_thread.dir = user_dir
                self.finger_thread.wave_received.connect(self.display_sensor_waveform)
                self.finger_thread.report_received.connect(self.show_sensors_report)
                self.finger_thread.start()
                print("指夹数据采集线程已启动。")
            else:
                print("指夹数据采集线程已在运行。")
        else:
            print("指夹串口未选择或初始化失败。")

    def show_sensors_report(self, report):
        # 直接使用原始报告内容，仅转换换行符
        processed_report = report.replace('\n', '<br>').replace('【', '<strong>【').replace('】', '】</strong>')
        
        html_report = f"""
        <div class="report-section pulse-section">
            <h3>脉象分析报告 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</h3>
            <div class="vital-signs-text">
                {processed_report}
            </div>
        </div>
        <hr>
        """
        self.diagnosis_report.append(html_report)
        self.save_diagnosis_to_chat_history("脉诊", report.strip())
        self.status_bar.showMessage("智能脉诊已完成")

    def stop_sensors(self):
        if self.finger_thread is not None:
            self.finger_thread.stop()
            self.cnt=0
            self.x_time=[]
            self.y_wave=[]
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
        
    def start_ai_diagnosis(self):
        """启动AI综合诊断"""
        # 确保聊天模型已初始化
        chat_model = self.get_or_create_chat_model()
        
        # 强制重置消息历史
        chat_model.messages = []
        if chat_model.system_prompt:
            chat_model.messages.append({"role": "system", "content": chat_model.system_prompt})
        
        # 收集诊断报告
        selected_user = self.user_combo.currentText()
        user_dir = os.path.join(self.patient_list_dp, selected_user)
        history_files = [f for f in os.listdir(user_dir) if f.startswith("chat_history")]
        
        # 合并所有诊断内容
        combined_content = []
        for fname in history_files:
            with open(os.path.join(user_dir, fname), 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_content.extend([msg["content"] for msg in data])
        
        # 添加诊断报告作为第一条用户消息
        if combined_content:
            chat_model.messages.append({
                "role": "user",
                "content": "请根据以下诊断数据进行综合分析：\n" + "\n".join(combined_content)
            })
        else:
            QtWidgets.QMessageBox.warning(self, '无数据', '没有可用的诊断报告')
            return
        
        # 获取回答
        try:
            response = chat_model.get_answer("")  # 空问题，直接使用现有消息
            self.show_ai_response(response)
        except Exception as e:
            print(f"综合诊断错误: {str(e)}")
            QtWidgets.QMessageBox.critical(self, '错误', f'综合诊断失败: {str(e)}')

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
        """执行舌诊分析"""
        # 调用舌头诊断函数
        color_report, coating_report, cancer_report, tongue_annotated, diagnosis, treatment = tongue_diagnose_sum(image_path, is_original_frame)
        
        # 使用最新的裁剪图像路径
        display_image_path = self.latest_crop_tongue_path if hasattr(self, 'latest_crop_tongue_path') and self.latest_crop_tongue_path else image_path
        
        # 将图像转换为base64格式以嵌入HTML
        try:
            with open(display_image_path, "rb") as img_file:
                import base64
                from pathlib import Path
                img_format = Path(display_image_path).suffix[1:]  # 获取扩展名，去掉点
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                img_src = f"data:image/{img_format};base64,{img_data}"
        except Exception as e:
            print(f"图像嵌入错误: {e}")
            img_src = ""
        
        # 预处理所有文本（修复反斜杠问题）
        diagnosis_processed = diagnosis.replace('\n', '<br>')
        treatment_processed = treatment.replace('\n', '<br>')
        color_processed = color_report.replace('\n', '<br>')
        coating_processed = coating_report.replace('\n', '<br>')
        cancer_processed = cancer_report.replace('\n', '<br>')
        
        html_report = f"""
        <div class="report-section tongue-section">
            <h3>舌诊分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>
            <div class="tongue-content">
                <div class="tongue-image">
                    <img src="{img_src}" alt="舌像分析图">
                </div>
                <div class="diagnosis-text">
                    <div class="tongue-parameters">
                        <h4>舌色分析</h4>
                        <p>{color_processed}</p>
                        <h4>舌苔分析</h4>
                        <p>{coating_processed}</p>
                        <h4>异常检测</h4>
                        <p>{cancer_processed}</p>
                    </div>
                    <div class="diagnosis-summary">
                        <h4>诊断结论</h4>
                        <p>{diagnosis_processed}</p>
                        <h4>治疗建议</h4>
                        <p>{treatment_processed}</p>
                    </div>
                </div>
            </div>
        </div>
        <hr>
        """
        
        # 添加到诊断报告
        self.diagnosis_report.append(html_report)
        
        # 同时将诊断结果保存到聊天历史
        report_text = f"""
诊断结论: {diagnosis}
治疗建议: {treatment}
舌色分析: {color_report}
舌苔分析: {coating_report}
异常检测: {cancer_report}
        """
        self.save_diagnosis_to_chat_history("舌诊", report_text.strip())
        
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
        
        # 将图像转换为base64格式以嵌入HTML
        try:
            with open(annotated_img_path, "rb") as img_file:
                import base64
                from pathlib import Path
                img_format = Path(annotated_img_path).suffix[1:]
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                img_src = f"data:image/{img_format};base64,{img_data}"
        except Exception as e:
            print(f"图像嵌入错误: {e}")
            img_src = ""
        
        # 预处理诊断报告（先处理换行符）
        processed_report = diagnosis_report.replace('\n', '<br>').replace('【', '<strong>【').replace('】', '】</strong>')
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 确保已导入datetime
        
        html_report = f"""
        <div class="report-section face-section">
            <h3>面诊分析报告 - {timestamp}</h3>
            <div class="face-content">
                <div class="face-image">
                    <img src="{img_src}" alt="面部分析图">
                </div>
                <div class="diagnosis-text">
                    {processed_report}  <!-- 使用预处理后的变量 -->
                </div>
            </div>
        </div>
        <hr>
        """
        
        # 添加到诊断报告
        self.diagnosis_report.append(html_report)
        
        # 同时将诊断结果保存到聊天历史
        self.save_diagnosis_to_chat_history("面诊", diagnosis_report.strip())
        
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


    def export_report(self):
        """导出诊断报告为HTML"""
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            QMessageBox.warning(self, '无用户', '请先添加用户。')
            return
        
        try:
            # 获取完整HTML内容
            html_content = self.diagnosis_report.toHtml()
            
            # 增强HTML结构
            enhanced_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{selected_user}的诊断报告</title>
                <style>
                    body {{ 
                        font-family: 'Microsoft YaHei', SimSun, sans-serif;
                        margin: 30px;
                        line-height: 1.6;
                    }}
                    h2 {{ 
                        color: #2c3e50;
                        border-bottom: 2px solid #3498db;
                        padding-bottom: 10px;
                    }}
                    .report-section {{
                        margin: 20px 0;
                        padding: 20px;
                        background: #f8f9fa;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    img {{
                        max-width: 80%;
                        display: block;
                        margin: 15px auto;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }}
                    .timestamp {{
                        color: #666;
                        font-size: 0.9em;
                        margin-bottom: 5px;
                    }}
                    /* 新增面诊专属样式 */
                    .face-content {{
                        display: grid;
                        grid-template-columns: 300px 1fr;
                        gap: 25px;
                        margin-top: 20px;
                    }}
                    
                    .face-image img {{
                        max-width: 100%;
                        border: 1px solid #ddd;
                        border-radius: 6px;
                        padding: 5px;
                    }}
                    
                    .diagnosis-text {{
                        white-space: pre-wrap;  /* 保留换行 */
                        font-family: 'Microsoft YaHei', sans-serif;
                        font-size: 16px;
                        line-height: 1.8;
                        color: #444;
                    }}
                    
                    .diagnosis-text strong {{
                        color: #2c3e50;
                        font-weight: 600;
                    }}
                    
                    .diagnosis-text br {{
                        content: "";
                        display: block;
                        margin-bottom: 12px;
                    }}
                </style>
            </head>
            <body>
                <h2>{selected_user}的智能中医诊断报告</h2>
                {html_content}
            </body>
            </html>
            """
            
            # 创建文件保存对话框
            file_dialog = QFileDialog()
            file_dialog.setDefaultSuffix("html")
            default_name = f"{selected_user}_诊断报告_{datetime.now().strftime('%Y%m%d')}.html"
            file_path, _ = file_dialog.getSaveFileName(
                self,
                "保存诊断报告",
                os.path.join(os.path.expanduser("~"), "Desktop", default_name),
                "HTML文件 (*.html);;所有文件 (*)"
            )
            
            if not file_path:  # 用户取消选择
                return
            
            # 保存文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(enhanced_html)
            
            # 显示成功提示
            QMessageBox.information(
                self, 
                '导出成功', 
                f'HTML报告已保存到：\n{file_path}\n\n'
                '可以用任意浏览器打开查看。'
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                '导出失败', 
                f'无法保存文件：{str(e)}\n'
                '请检查：\n'
                '1. 是否有写入权限\n'
                '2. 磁盘空间是否充足'
            )
            print(f"导出错误: {traceback.format_exc()}")

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
            print(f"显示摄像头帧d出错: {e}")

    
    def display_sensor_waveform(self, data):
        """显示脉搏数据"""
        try:
            if self.finger_start_time is None:
                self.finger_start_time = datetime.now()
            current_time = datetime.now()
            time_diff = (current_time - self.finger_start_time).total_seconds()
            relative_time = time_diff % self.time_window
            cnt = time_diff // self.time_window
            if cnt>self.cnt:
                self.cnt=cnt
                self.x_time = []
                self.y_wave = []
                
            for _, value in data:
                self.x_time.append(relative_time)
                self.y_wave.append(value)

            self.graphWidget.clear()  # 清除旧曲线
            self.graphWidget.plot(self.x_time, self.y_wave, pen=pg.mkPen(color='b', width=2))

        except Exception as e:
            print(f"接受脉搏数据出错: {e}")
        
    
    
    def show_guidance(self, message):
        """显示检测引导提示"""
        print("调用show_guidance函数")
        
            
        # 更新状态栏
        status_msg = message
        self.status_bar.showMessage(status_msg)
            
            

    def handle_new_crop_image(self, path):
        """处理新的裁剪舌头图像"""
        self.latest_crop_tongue_path = path
        print(f"新裁剪舌头图像已保存: {path}")
        
        # 如果需要在界面上显示裁剪图像，可以在这里添加显示代码
        # 例如:
        # pixmap = QPixmap(path)
        # self.some_label.setPixmap(pixmap.scaled(...))

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

    def save_diagnosis_to_chat_history(self, diagnosis_type, report_text):
        """将诊断结果保存到聊天历史记录中"""
        selected_user = self.user_combo.currentText()
        if selected_user == "无用户，请添加":
            return
        
        user_dir = os.path.join(self.patient_list_dp, selected_user)
        history_file = os.path.join(user_dir, "chat_history.json")
        
        # 构建消息
        message = {
            "role": "user",
            "content": f"[{diagnosis_type}报告] {report_text}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 读取现有历史记录
        history = []
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                history = []
        
        # 添加新消息
        history.append(message)
        
        # 保存更新后的历史记录
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        # 如果聊天模型已经初始化，更新其历史记录
        if hasattr(self, 'cloud_chat') and self.current_chat_history_file == history_file:
            self.cloud_chat.load_history()
        elif hasattr(self, 'local_chat') and self.current_chat_history_file == history_file:
            self.local_chat.load_history()
        
        # 在聊天历史UI中显示诊断报告
        # self.chat_history.append(f'''
        # <div style="background: #e6f7ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
        #     <p><span style="color: #888; font-size: 0.8em;">{message['timestamp']}</span></p>
        #     <p><b>[{diagnosis_type}报告]</b></p>
        #     <p>{report_text.replace(chr(10), '<br>')}</p>
        # </div>
        # ''')

    def show_ai_response(self, response):
        """显示AI回复"""
        # 清空旧内容
        self.chat_history.clear()
        
        # 设置Markdown支持
        self.chat_history.setOpenExternalLinks(True)
        self.chat_history.document().setDefaultStyleSheet("""
            pre {
                white-space: pre-wrap;
                background: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
            }
            code {
                background: #f8f9fa;
                padding: 2px 4px;
            }
            blockquote {
                border-left: 3px solid #3498db;
                margin: 5px 0;
                padding-left: 10px;
                color: #666;
            }
        """)
        
        # 显示渲染后的HTML
        self.chat_history.setHtml(response)
        
        # 自动滚动到底部
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


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
