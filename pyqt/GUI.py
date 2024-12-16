# -*- coding: UTF-8 -*-
# 这个文件运行就可以打开运行界面了，其他的文件目录都是yolo模型的，主要都在这个文件里写就好
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import torchvision
import cv2

from ultralytics import YOLO
import os
import serial
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt

# 检测结果类别对应的诊断报告
class_labels = {
    0: "您的舌质呈现粉红色，这通常与健康的舌象相符，表明您的身体状况良好，气血充足。粉红舌通常反映出良好的生理状态，然而，如果舌质偏红，则可能提示体内存在热症，需警惕潜在的炎症或感染情况。建议定期关注身体其他症状，保持健康的生活方式。",
    
    1: "您的舌苔薄白，通常表明体内没有明显的病理变化，可能处于健康状态。然而，薄白舌也可能提示气血不足或体内寒气较重，建议注意饮食调理，适当增加营养摄入，保持身体温暖，避免寒凉食物的过量摄入。",
    
    2: "您的舌苔厚白，这可能指示体内存在寒湿或痰湿，通常与消化系统功能障碍有关。厚白舌常见于脾胃虚弱、消化不良等情况，建议您关注饮食习惯，避免油腻和生冷食物，同时可以考虑适当的中药调理，以增强脾胃功能。",
    
    3: "您的舌苔厚黄，这通常表示体内有湿热，可能伴随发热、口渴、便秘等症状。厚黄舌常见于感染、炎症或消化系统疾病。建议您保持充足的水分摄入，避免辛辣刺激食物，同时可以考虑咨询专业医生进行进一步检查和调理。",
    
    4: "您的舌苔灰黑，这是一种较为严重的病理变化，可能与严重的感染、长期疾病、药物中毒或内脏器官的严重病变有关。灰黑舌通常提示体内存在较大的病理变化，建议您尽快就医，进行详细检查，以便及时发现并处理潜在的健康问题。"
}



class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.start_camera = False  # 标志位，用于控制摄像头的开启和关闭
        self.detection_reports = []  # 用于存储检测报告的列表
        self.patient_id = 0  # 病人编号
        self.ser = None  # 串口对象
        self.received_data = []  # 存储接收到的脉搏数据
        self.timestamps = []  # 存储时间戳
        self.detecting = False  # 检测状态标志
        
        
    def setupUi(self, MainWindow): # 设置界面的组件，包括主窗口、按钮、标签等
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1128, 1009)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(20, 10, 93, 28))
        self.pushButton.setObjectName("pushButton")
        # self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton_2.setGeometry(QtCore.QRect(160, 10, 93, 28))
        # self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(290, 10, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label1 = QtWidgets.QTextBrowser(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(20, 60, 1071, 71))
        self.label1.setObjectName("label1")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(40, 190, 481, 421))
        self.label2.setObjectName("label2")
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(600, 200, 461, 381))
        self.label3.setObjectName("label3")
        MainWindow.setCentralWidget(self.centralwidget)
        
        # 添加开始检测按钮
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(430, 10, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setText("开始检测")
        # self.pushButton_4.clicked.connect(self.toggleDetection)
        MainWindow.setCentralWidget(self.centralwidget)
        
        
        # 显示检测报告的文本框
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(20, 650, 1071, 211))
        self.textBrowser.setObjectName("textBrowser")
        
        MainWindow.setCentralWidget(self.centralwidget)
        
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1128, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        # 点击响应函数
        self.pushButton.clicked.connect(self.uploadImage)
        # self.pushButton_2.clicked.connect(self.showEnvironment)
        self.pushButton_3.clicked.connect(self.startProgram)
        self.pushButton_4.clicked.connect(self.startDetection)
        # self.image_path = ''

        self.model = YOLO('./runs/detect/train/weights/best.pt')  # 加载模型
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        
        # 创建保存图片和标签的文件夹
        self.images_dir = "tongue_images"
        self.labels_dir = "tongue_labels"
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def retranslateUi(self, MainWindow):# 设置界面各个组件的文本内容。
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "上传图片"))
        # self.pushButton_2.setText(_translate("MainWindow", "显示环境"))
        self.pushButton_3.setText(_translate("MainWindow", "执行检测"))
        self.pushButton_4.setText(_translate("MainWindow", "开始检测"))
        self.label2.setText(_translate("MainWindow", "TextLabel"))
        self.label3.setText(_translate("MainWindow", "TextLabel"))
        
        

    def uploadImage(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, '选择图片', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        self.image_path = image_path
        if image_path:
            # 在这里添加加载图片的逻辑，例如显示图片到label2
            pixmap = QtGui.QPixmap(image_path)
            self.label2.setPixmap(pixmap)
            self.label2.setScaledContents(True)
            
            
    # 显示设备环境的函数，不需要      
    def showEnvironment(self):
        pytorch_version = torch.__version__
        torchvision_version = torchvision.__version__
        self.label1.setText(f"PyTorch Version: {pytorch_version}\n"
                            f"Torchvision Version: {torchvision_version}")

    # 上传图片文件，点击执行检测按钮对应的函数
    def startProgram(self):

        self.label1.setText(self.image_path)
        
        results = self.model(self.image_path)
        self.generate_report(results)
        # self.save_images_and_labels(frame, results)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        # 将图像数据转换为QImage格式
        height, width, channel = annotated_frame.shape
        bytes_per_line = 3 * width
        qimage = QtGui.QImage(annotated_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        # 将QImage转换为QPixmap
        pixmap = QtGui.QPixmap.fromImage(qimage)

        self.label3.setPixmap(pixmap)
        self.label3.setScaledContents(True)
        
    # 开始检测按钮对应的函数
    def startDetection(self):
        if not self.start_camera:  # 如果摄像头未开启，则开启摄像头
            self.start_camera = True
            self.pushButton_4.setText("停止检测")
            self.capture_frame()  # 开始从摄像头画面中截图
            self.patient_id += 1  # 增加病人编号
        else:  # 如果摄像头已开启，则停止检测
            self.start_camera = False
            self.pushButton_4.setText("开始检测")
            self.cap.release()  # 释放摄像头
            self.clearReports()  # 清空报告
            
    
    # 实时检测主函数
    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.label2.setPixmap(self.convert_cv_qt(frame))  # 显示摄像头画面
            if self.start_camera:  # 如果处于开始检测状态，则进行检测
                results = self.model(frame)  # 直接使用摄像头画面进行检测
                # print("printresults",results)
                # # 提取检测结果
                # for result in results:
                #     boxes = result.boxes.xyxy  # 边界框坐标
                #     scores = result.boxes.conf  # 置信度分数
                #     classes = result.boxes.cls  # 类别索引
                    
                #     # 如果有类别名称，可以通过类别索引获取
                #     class_names = [self.model.names[int(cls)] for cls in classes]
                    
                #     # 打印检测结果
                #     for box, score, class_name in zip(boxes, scores, class_names):
                #         print(f"Class: {class_name}, Score: {score:.2f}, Box: {box}")
                        
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                height, width, channel = annotated_frame.shape
                bytes_per_line = 3 * width
                qimage = QtGui.QImage(annotated_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimage)
                self.label3.setPixmap(pixmap)
                self.label3.setScaledContents(True)
                
                # 获取检测结果，并生成报告
                self.generate_report(results)
                self.save_images_and_labels(frame, results)
        QtCore.QTimer.singleShot(10, self.capture_frame)  # 每隔10毫秒从摄像头画面中截图
        
    # 工具函数：用于在qt显示图像
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = QtGui.QPixmap.fromImage(convert_to_Qt_format)
        return p    
    
    # 工具函数：用于生成舌象诊断报告
    def generate_report(self, results):
        for result in results:
            class_ids = result.boxes.cls.numpy()  # 获取类别索引数组
            for class_id in class_ids:
                class_name = self.model.names[int(class_id)]
                diagnosis = class_labels.get(class_id, "未知类别")
                report = f"\n{diagnosis}"
                if report not in self.detection_reports:  # 避免重复添加相同的报告
                    self.detection_reports.append(report)
        # self.updateTextBrowser()

    # 工具函数：用于更新舌象诊断报告，待调试
    # TODO：现在有容易把人脸检测成灰黑色舌头的问题，想办法让检测报告只输出正确的检测结果
    # def updateTextBrowser(self):
    #     report = "\n".join(self.detection_reports)  # 将所有报告添加到文本框中
    #     self.textBrowser.setPlainText(report)  # 更新文本框内容

    # 工具函数：每次点击开始检测，清除上一次检测报告
    def clearReports(self):
        self.detection_reports.clear()  # 清空报告列表
        self.textBrowser.setPlainText("")  # 清空文本框内容
        
    # 用于保存舌诊图像和标签，TODO：还没跑通，待改
    def save_images_and_labels(self, image, results):
        # 检查是否有检测结果
        if len(results) == 0:
            return
        
        # 保存舌象图片
        image_path = os.path.join(self.images_dir, f"patient_{self.patient_id}.png")
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # 保存诊断结果编号
        label_path = os.path.join(self.labels_dir, f"patient_{self.patient_id}.txt")
        for result in results:
            class_ids = result.boxes.cls.numpy()  # 获取类别索引数组
            with open(label_path, 'w') as f:
                for class_id in class_ids:
                    f.write(f"{int(class_id)}\n")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()     
    ui = Ui_MainWindow()             
    ui.setupUi(MainWindow1)
    MainWindow1.show()
    sys.exit(app.exec_())
