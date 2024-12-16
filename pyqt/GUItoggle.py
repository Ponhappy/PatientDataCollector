# 这个文件是我之前测试读取串口的函数！ 还没跑通，可以参考

# -*- coding: UTF-8 -*-
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import torch
import torchvision
import cv2
from ultralytics import YOLO
import serial
import csv
import time
from datetime import datetime



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

    def setupUi(self, MainWindow):
       

        # 添加开始/停止检测按钮
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(430, 10, 93, 28))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setText("开始检测")
        self.pushButton_4.clicked.connect(self.toggleDetection)


     

    def toggleDetection(self):
        if not self.detecting:
            # 开始检测
            self.start_camera = True
            self.patient_id += 1  # 增加病人编号
            self.startSerial()  # 启动串口读取
            self.pushButton_4.setText("停止检测")
            self.capture_frame()  # 开始从摄像头画面中截图
        else:
            # 停止检测
            self.start_camera = False
            self.stopSerial()  # 停止串口读取并保存数据
            self.pushButton_4.setText("开始检测")
            self.cap.release()  # 释放摄像头

    def startSerial(self):
        self.ser = serial.Serial('COM3', baudrate=38400, timeout=1, bytesize=8, stopbits=1, parity='N')
        if not self.ser.is_open:
            self.ser.open()
        self.ser.write(bytes([0x8A]))  # 发送单个字节 0x8A 启动设备
        time.sleep(0.5)  # 等待设备响应的时间
        self.readSerialData()  # 开始读取串口数据
        self.detecting = True

    def readSerialData(self):
        if self.ser.in_waiting > 0:  # 检查串口是否有数据
            raw_data = self.ser.read(self.ser.in_waiting)  # 读取所有可用的数据
            hex_data = raw_data.hex().upper()  # 转换为大写的十六进制字符串
            formatted_data = " ".join([hex_data[i:i + 2] for i in range(0, len(hex_data), 2)])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            self.received_data.append([timestamp, formatted_data])
            self.timestamps.append(timestamp)
            print(formatted_data)
        QtCore.QTimer.singleShot(10, self.readSerialData)  # 每隔10毫秒读取一次串口数据

    def stopSerial(self):
        if self.ser:
            self.ser.close()
            print("Serial connection closed.")
            self.saveSerialData()  # 保存串口数据到CSV文件
            self.detecting = False

    def saveSerialData(self):
        csv_filename = f"pulse_data_patient_{self.patient_id}.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Data"])  # 写入表头
            for data in self.received_data:
                writer.writerow(data)  # 写入每一行数据
        print(f"Pulse data saved to {csv_filename}")

   

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow1 = QMainWindow()     
    ui = Ui_MainWindow()             
    ui.setupUi(MainWindow1)
    MainWindow1.show()
    sys.exit(app.exec_())