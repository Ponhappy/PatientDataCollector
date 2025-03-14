import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore
import serial
import csv
import datetime
import time

def save_wrist_pulse(file_path, port):
    # 设置串口参数
    baudrate = 115200
    timeout = 1
    # 初始化串口
    ser = serial.Serial(port, baudrate, timeout=timeout)
    print("串口无问题")
    # 创建一个 CSV 文件并写入表头
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Waveform", "Heartbeat", "Heart Rate", "HRV"])
    print("文件已经创造")
    start_time = time.time()
    while True:
        if time.time() - start_time > 30:  # 检查是否已经过了 30 秒
            break
        flag = ser.readline()
        if flag and flag[-1] == 0xff:  # 检查最后一个字节是否为 0xff
            print("无效输入")
            break
        try:
            # 读取一行数据
            data = ser.readline().decode('ascii').strip()
            # 检查数据格式，确保包含四个由逗号分隔的值
            if data:
                parts = data.split(',')
                if len(parts) == 4:
                    # 获取当前时间戳
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # 将数据和时间戳一起写入 CSV 文件
                    with open(file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([timestamp, int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
        except KeyboardInterrupt:
            print("程序中断")
            break
    # 关闭串口
    ser.close()




