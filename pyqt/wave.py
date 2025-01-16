# wave.py

import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout,QApplication
from PyQt5.QtCore import QTimer, QDateTime
import serial

class wrist_PlotWidget(QWidget):
    def __init__(self, baudrate=115200, duration=30000):
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
        self.setFixedSize(1000, 300)  # 宽度为 1000 像素，高度为 300 像素
        # 初始数据
        self.data = np.array([0])
        # 初始横坐标，这里假设横坐标范围是从 0 到 100，共 100 个点
        self.x = np.linspace(0, 100, 100)
        # 启动定时器
        self.start_timer()

        self.port = None  # 初始时不设置串口
        self.baudrate = baudrate
        self.timeout = 1
        self.ser = None  # 串口对象

        self.duration = duration  # 定时器持续时间，单位为毫秒
        self.start_time = QDateTime.currentDateTime().toMSecsSinceEpoch()  # 记录定时器开始时间

    def start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start(100)  # 每 100 毫秒更新一次曲线

    def set_serial_port(self, port, baudrate=115200):
        # 仅记录串口信息，不打开串口
        self.port = port
        self.baudrate = baudrate
        print(f"wrist_PlotWidget 关联的串口: {self.port}")
        # 如果需要，可以添加串口数据的接收和绘图逻辑

    def timerEvent(self):
        # wrist_PlotWidget 仅负责绘图，不处理串口数据
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = wrist_PlotWidget(duration=5000)  # 设定持续时间为 5 秒
    widget.show()
    sys.exit(app.exec_())
