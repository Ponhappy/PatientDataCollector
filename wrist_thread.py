# wrist_thread.py

from PyQt5.QtCore import QThread, pyqtSignal
import serial
import csv
import time
from datetime import datetime
from pulse_diagnose_model.wrist_detect import save_wrist_pulse

hrv_values = []
received_data2 = []  # 存储 COM3 数据

def wrist_detect(file_path, port="COM3"):
    baudrate = 115200
    timeout = 1
    ser = serial.Serial(port, baudrate, timeout=timeout)
    print(f"正在从 {port} 读取数据...")
    # 创建 CSV 文件并写入表头
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Timestamp", "Waveform", "Heartbeat", "Heart Rate", "HRV"])

    start_time = time.time()
    while True:
        if time.time() - start_time > 30:  # 采集 30 秒
            break
        try:
            data = ser.readline().decode('ascii').strip()
            if data:
                parts = data.split(',')
                if len(parts) == 4:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    waveform, heartbeat, heart_rate, hrv = map(int, parts)

                    # 存入 received_data2
                    received_data2.append([timestamp, waveform, heartbeat, heart_rate, hrv])

                    # 写入 CSV
                    with open(file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([timestamp, waveform, heartbeat, heart_rate, hrv])

        except KeyboardInterrupt:
            print("程序中断")
            break
        except Exception as e:
            print(f"错误: {e}")
            continue

    ser.close()
    print(f"{port} 数据采集完成，数据已保存到 {file_path}")
    hrv_values = [row[4] for row in received_data2 if row[4] != 0]  # 提取 HRV 数据
    avg_hrv = sum(hrv_values) / len(hrv_values) if hrv_values else 0
    diagnosis_report = []
    if avg_hrv < 22:
        diagnosis_report.append(
            "心率变异性：偏低（平均HRV：{:.2f}）。交感神经活跃，可能导致心跳加速、血压升高、胃肠蠕动减慢等症状。".format(
                avg_hrv))
    elif avg_hrv > 120:
        diagnosis_report.append("心率变异性：偏高（平均HRV：{:.2f}）。可能存在心律失常。".format(avg_hrv))
    else:
        diagnosis_report.append("心率变异性：正常（平均HRV：{:.2f}）。".format(avg_hrv))
    return "\n".join(diagnosis_report)

class WristDataThread(QThread):
    data_received = pyqtSignal(str)  # 发送 HRV 诊断报告

    def __init__(self, serial_port="COM3", baudrate=38400, parent=None):
        super().__init__(parent)
        # 确保 serial_port 是字符串类型
        if isinstance(serial_port, int):
            self.serial_port = str(serial_port)  # 将传入的整数端口转换为字符串
        else:
            self.serial_port = serial_port
        self.baudrate = baudrate
        self.running = True
        self.csv_filename = "wrist_pulse_data.csv"

    def run(self):
        report = wrist_detect(self.csv_filename, port=self.serial_port)
        self.data_received.emit(report)  # 发送诊断报告
        print(report)
    def stop(self):
        self.running = False
        self.wait()
