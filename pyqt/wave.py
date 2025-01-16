import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow
import serial,datetime,openpyxl

class finger_PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.plot_widget = pg.PlotWidget()
        # 禁用鼠标交互
        self.plot_widget.setMouseEnabled(False, False)
        self.plot_widget.plotItem.setMenuEnabled(False)
        self.plot_widget.plotItem.setMouseEnabled(False, False)
        self.curve = self.plot_widget.plot(pen='y')  # 创建一个黄色的曲线
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)
        # 固定 PlotWidget 的大小
        self.setFixedSize(500, 300)  # 宽度为 500 像素，高度为 300 像素
        # 初始数据
        self.data = np.array([0])
        # 初始横坐标，这里假设横坐标范围是从 0 到 100，共 100 个点
        self.x = np.linspace(0, 100, 100)
        # 启动定时器
        self.start_timer()




    def start_timer(self):
        self.timer = self.startTimer(100)  # 每 100 毫秒更新一次曲线

    def timerEvent(self, event):
        if self.ser is not None:
            try:
                # 从串口读取数据
                data = self.ser.readline().decode('ascii').strip()
                if data:  # 确保读取到了数据
                    parts = data.split(',')
                    if len(parts) > 0:  # 确保数据格式正确
                        # 尝试将第一个部分转换为浮点数
                        heart_beat = float(parts[0])
                        print("heart_beat:", heart_beat)
                        new_data = np.array([heart_beat])
                        # 移除最旧的数据
                        self.data = np.append(self.data[1:], new_data)
                        # 用 np.nan 填充 self.data 直到其长度与 self.x 相同
                        while len(self.data) < len(self.x):
                            self.data = np.append(self.data, np.nan)
                        # 更新横坐标，将最旧的横坐标点移除，添加新的横坐标点
                        self.x = np.append(self.x[1:], self.x[-1] + 1)
                        self.curve.setData(self.x, self.data)  # 更新曲线，同时更新横坐标和纵坐标
                    else:
                        print("Invalid data format received from serial port.")
                else:
                    print("No data received from serial port.")
            except Exception as e:
                print(f"Error processing serial data: {e}")
        else:
            print("Serial port is not open.")





import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore
import serial


class wrist_PlotWidget(QWidget):
    def __init__(self, duration=30000):  # 默认持续时间为 10 秒（10000 毫秒）
        super().__init__()
        self.plot_widget = pg.PlotWidget()
        # 禁用鼠标交互
        self.plot_widget.setMouseEnabled(False, False)
        self.plot_widget.plotItem.setMenuEnabled(False)
        self.plot_widget.plotItem.setMouseEnabled(False, False)
        self.curve = self.plot_widget.plot(pen='y')  # 创建一个黄色的曲线
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)
        # 固定 PlotWidget 的大小
        self.setFixedSize(1000, 300)  # 宽度为 500 像素，高度为 300 像素
        # 初始数据
        self.data = np.array([0])
        # 初始横坐标，这里假设横坐标范围是从 0 到 100，共 100 个点
        self.x = np.linspace(0, 100, 100)
        # 启动定时器
        self.start_timer()

        self.port = 'COM5'  # 替换为你的串口号
        self.baudrate = 115200
        self.timeout = 1
        # 初始化串口
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except serial.SerialException as e:
            print(f"Failed to open serial port: {e}")
            self.ser = None

        self.duration = duration  # 定时器持续时间，单位为毫秒
        self.start_time = QtCore.QDateTime.currentDateTime().toMSecsSinceEpoch()  # 记录定时器开始时间


    def start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start(100)  # 每 100 毫秒更新一次曲线


    def timerEvent(self):
        if self.ser is not None:
            try:
                # 从串口读取数据
                data = self.ser.readline().decode('ascii').strip()
                if data:  # 确保读取到了数据
                    parts = data.split(',')
                    if len(parts) > 0:  # 确保数据格式正确
                        # 尝试将第一个部分转换为浮点数
                        Waveform=float(parts[0])
                        Heartbeat=float(parts[1])
                        Heart_Rate = float(parts[2])
                        HRV=float(parts[3])
                        print("Waveform:", Waveform)
                        print("Heartbeat:", Heartbeat)
                        print("Heart_Rate:", Heart_Rate)
                        print("HRV:", HRV)

                        new_data = np.array([Waveform])
                        # 移除最旧的数据
                        self.data = np.append(self.data[1:], new_data)
                        # 用 np.nan 填充 self.data 直到其长度与 self.x 相同
                        while len(self.data) < len(self.x):
                            self.data = np.append(self.data, np.nan)
                        # 更新横坐标，将最旧的横坐标点移除，添加新的横坐标点
                        self.x = np.append(self.x[1:], self.x[-1] + 1)
                        self.curve.setData(self.x, self.data)  # 更新曲线，同时更新横坐标和纵坐标
                    else:
                        print("Invalid data format received from serial port.")
                else:
                    print("No data received from serial port.")
                # 检查是否达到指定时间
                current_time = QtCore.QDateTime.currentDateTime().toMSecsSinceEpoch()
                if (current_time - self.start_time) >= self.duration:
                    self.timer.stop()  # 停止定时器
                    self.timer.deleteLater()  # 清理定时器资源
                    print("Timer stopped.")
            except Exception as e:
                print(f"Error processing serial data: {e}")
        else:
            print("Serial port is not open.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = wrist_PlotWidget(duration=5000)  # 设定持续时间为 5 秒
    widget.show()
    sys.exit(app.exec_())






