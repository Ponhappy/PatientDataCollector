# finger_thread.py

from PyQt5.QtCore import QThread, pyqtSignal
import serial
import csv
import time
from datetime import datetime
from finger_detect import save_finger_pulse

class FingerDataThread(QThread):
    data_received = pyqtSignal(float)  # 定义一个信号，发送脉搏数据

    def __init__(self, serial_port, baudrate=115200, parent=None):
        super().__init__(parent)
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.running = True
        self.csv_filename = None

    def run(self):
        save_finger_pulse(self.csv_filename,self.serial_port)
        # try:
        #     ser = serial.Serial(self.serial_port, self.baudrate, timeout=1, bytesize=8, stopbits=1, parity='N')
        #     print(f"指夹串口 {self.serial_port} 已打开。")
        # except serial.SerialException as e:
        #     print(f"无法打开指夹串口 {self.serial_port}: {e}")
        #     return

        # if self.csv_filename:
        #     csv_file = open(self.csv_filename, mode='w', newline='')
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(["Timestamp", "Pulse Value"])  # 写入表头
        # else:
        #     csv_writer = None

        # try:
        #     while self.running:
        #         if ser.in_waiting > 0:
        #             line = ser.readline().decode('ascii').strip()
        #             if line:
        #                 try:
        #                     pulse_value = float(line)
        #                     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        #                     self.data_received.emit(pulse_value)  # 发送信号
        #                     if csv_writer:
        #                         csv_writer.writerow([timestamp, pulse_value])
        #                 except ValueError:
        #                     print(f"数据格式错误: {line}")
        #         time.sleep(0.1)  # 调整采样频率
        # finally:
        #     ser.close()
        #     if csv_writer:
        #         csv_file.close()
        #     print(f"指夹串口 {self.serial_port} 已关闭。")

    def stop(self):
        self.running = False
        self.wait()
