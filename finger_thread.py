# finger_thread.py

from PyQt5.QtCore import QThread, pyqtSignal
import serial
import csv
import time
from datetime import datetime
from finger_detect import save_finger_pulse
import serial
import csv
import time
from datetime import datetime


def finger_detect(port='COM4', baudrate=38400, packet_limit=30):
    def to_signed_byte(byte):
        return byte - 256 if byte > 127 else byte

    # 打开串口连接
    ser = serial.Serial('COM4', baudrate=38400, timeout=1, bytesize=8, stopbits=1, parity='N')

    # 确保串口打开
    if not ser.is_open:
        ser.open()

    # 向串口发送启动命令 (0x8A)
    ser.write(bytes([0x8A]))  # 发送单个字节 0x8A 启动设备
    time.sleep(0.5)  # 等待设备响应的时间

    # 存储接收到的原始数据
    received_data = []
    timestamps = []  # 存储时间戳

    # 读取数据并保存为 CSV 文件
    try:
        packet_count = 0
        while packet_count < 30:  # 采集80个回合
            if ser.in_waiting >= 88:  # 检查串口是否有足够的数据
                # 读取一包数据
                raw_data = ser.read(88)  # 读取88个字节
                hex_data = raw_data.hex().upper()  # 将原始数据转换为十六进制字符串

                # 在每个字节之间添加空格
                formatted_data = ' '.join([hex_data[i:i + 2] for i in range(0, len(hex_data), 2)])

                # 时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                # 保存数据（时间戳，格式化后的原始数据）
                received_data.append([timestamp, formatted_data])
                timestamps.append(timestamp)

                # 实时输出数据
                print(f"Packet {packet_count + 1}: {formatted_data}")

                packet_count += 1

            # 为了避免高CPU占用，可以适当休息一下
            time.sleep(0.1)

    finally:
        # 关闭串口连接
        ser.close()
        print("Serial connection closed.")

    # 将数据保存为 CSV 文件
    csv_filename = "pulse_data.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Raw Data"])  # 写入表头
        for data in received_data:
            writer.writerow(data)  # 写入每一行数据

    print(f"Pulse data saved to {csv_filename}")

    # 解码数据
    decoded_data = []

    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            timestamp = row[0]  # 时间戳
            raw_data = row[1]  # 原始数据（十六进制字符串，每个字节用空格分隔）

            # 将十六进制字符串转换为字节列表
            byte_list = [int(x, 16) for x in raw_data.split()]

            # 解码数据
            decoded_packet = {
                "timestamp": timestamp,  # 时间戳
                "acdata": [to_signed_byte(byte) for byte in byte_list[1:65]],  # 心律波形数据
                "heart_rate": byte_list[65],  # 心率
                "spo2": byte_list[66],  # 血氧
                "bk": byte_list[67],  # 微循环
                "fatigue_index": byte_list[68],  # 疲劳指数
                "reserved_1": byte_list[69],  # 保留数据
                "systolic_pressure": byte_list[70],  # 收缩压
                "diastolic_pressure": byte_list[71],  # 舒张压
                "cardiac_output": byte_list[72],  # 心输出
                "peripheral_resistance": byte_list[73],  # 外周阻力
                "rr_interval": byte_list[74],  # RR间期
                "sdnn": byte_list[75],  # SDNN
                "rmssd": byte_list[76],  # RMSSD
                "nn50": byte_list[77],  # NN50
                "pnn50": byte_list[78],  # pNN50
                "rra": byte_list[79:85],  # RR间期相关数据
                "reserved_2": byte_list[85:87],  # 保留数据
            }

            # 保存解码后的数据
            decoded_data.append(decoded_packet)

    # 将解码后的数据保存到 CSV 文件
    output_csv_filename = "decoded_pulse_data.csv"
    with open(output_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow([
            "Timestamp",
            "AC Data",  # 心律波形数据
            "Heart Rate",  # 心率
            "SpO2",  # 血氧
            "BK",  # 微循环
            "Fatigue Index",  # 疲劳指数
            "Reserved 1",  # 保留数据
            "Systolic Pressure",  # 收缩压
            "Diastolic Pressure",  # 舒张压
            "Cardiac Output",  # 心输出
            "Peripheral Resistance",  # 外周阻力
            "RR Interval",  # RR间期
            "SDNN",  # SDNN
            "RMSSD",  # RMSSD
            "NN50",  # NN50
            "pNN50",  # pNN50
            "RRA",  # RR间期相关数据
            "Reserved 2",  # 保留数据
        ])
        # 写入数据
        for packet in decoded_data:
            writer.writerow([
                packet["timestamp"],
                " ".join(map(str, packet["acdata"])),  # 心律波形数据
                packet["heart_rate"],  # 心率
                packet["spo2"],  # 血氧
                packet["bk"],  # 微循环
                packet["fatigue_index"],  # 疲劳指数
                packet["reserved_1"],  # 保留数据
                packet["systolic_pressure"],  # 收缩压
                packet["diastolic_pressure"],  # 舒张压
                packet["cardiac_output"],  # 心输出
                packet["peripheral_resistance"],  # 外周阻力
                packet["rr_interval"],  # RR间期
                packet["sdnn"],  # SDNN
                packet["rmssd"],  # RMSSD
                packet["nn50"],  # NN50
                packet["pnn50"],  # pNN50
                " ".join(map(str, packet["rra"])),  # RR间期相关数据
                " ".join(map(str, packet["reserved_2"])),  # 保留数据
            ])

    print(f"Decoded data saved to {output_csv_filename}")

    # 读取解码后的数据
    decoded_csv_filename = "decoded_pulse_data.csv"
    decoded_data = []

    with open(decoded_csv_filename, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取表头
        for row in reader:
            # 解析每一行数据
            decoded_packet = {
                "timestamp": row[0],  # 时间戳
                "acdata": list(map(int, row[1].split())),  # 心律波形数据
                "heart_rate": int(row[2]),  # 心率
                "spo2": int(row[3]),  # 血氧
                "bk": int(row[4]),  # 微循环
                "fatigue_index": int(row[5]),  # 疲劳指数
                "reserved_1": int(row[6]),  # 保留数据
                "systolic_pressure": int(row[7]),  # 收缩压
                "diastolic_pressure": int(row[8]),  # 舒张压
                "cardiac_output": int(row[9]),  # 心输出
                "peripheral_resistance": int(row[10]),  # 外周阻力
                "rr_interval": int(row[11]),  # RR间期
                "sdnn": int(row[12]),  # SDNN
                "rmssd": int(row[13]),  # RMSSD
                "nn50": int(row[14]),  # NN50
                "pnn50": int(row[15]),  # pNN50
                "rra": list(map(int, row[16].split())),  # RR间期相关数据
                "reserved_2": list(map(int, row[17].split())),  # 保留数据
            }
            decoded_data.append(decoded_packet)

    # 剔除所有关键指标为0的错误数据包
    valid_data = [
        packet for packet in decoded_data
        if (packet["heart_rate"] != 0 and
            packet["spo2"] != 0 and
            packet["bk"] != 0)
        # packet["diastolic_pressure"] != 0 and
        # packet["cardiac_output"] != 0)
    ]

    # 计算有效数据的平均值
    def calculate_average(values):
        return sum(values) / len(values) if values else 0

    avg_heart_rate = calculate_average([packet["heart_rate"] for packet in valid_data])
    avg_spo2 = calculate_average([packet["spo2"] for packet in valid_data])
    avg_bk = calculate_average([packet["bk"] for packet in valid_data])
    avg_fatigue = calculate_average([packet["fatigue_index"] for packet in valid_data])
    avg_systolic_pressure = calculate_average([packet["systolic_pressure"] for packet in valid_data])
    avg_diastolic_pressure = calculate_average([packet["diastolic_pressure"] for packet in valid_data])
    avg_hrv = calculate_average([packet["rr_interval"] for packet in valid_data])  # 假设rr_interval为心率变异性

    # 生成诊断报告
    diagnosis_report = []

    # 心率诊断
    if avg_heart_rate < 60:
        diagnosis_report.append(
            "心率：心动过缓（平均心率：{:.2f} bpm）。可能存在头晕、乏力、倦怠、精神差的症状。".format(avg_heart_rate))
    elif avg_heart_rate > 100:
        diagnosis_report.append(
            "心率：心动过速（平均心率：{:.2f} bpm）。可能出现心慌、气短、乏力等症状。".format(avg_heart_rate))
    else:
        diagnosis_report.append("心率：正常（平均心率：{:.2f} bpm）。".format(avg_heart_rate))

    # 血氧饱和度诊断
    if avg_spo2 < 95:
        diagnosis_report.append(
            "血氧饱和度：较低（平均血氧：{:.2f}%）。可能出现呼吸不畅、四肢乏力、头晕及胸闷等症状。".format(avg_spo2))
    else:
        diagnosis_report.append("血氧饱和度：正常（平均血氧：{:.2f}%）。".format(avg_spo2))

    # 微循环诊断
    if avg_bk < 70:
        diagnosis_report.append(
            "微循环：血管指数过低（平均微循环：{:.2f}）。处于亚健康状态，容易出现疲乏无力、情绪低落、睡眠质量差等症状。".format(
                avg_bk))
    elif 70 <= avg_bk < 79:
        diagnosis_report.append(
            "微循环：血管指数较低（平均微循环：{:.2f}）。处于亚健康状态，容易出现疲乏无力、情绪低落、睡眠质量差等症状。".format(
                avg_bk))
    else:
        diagnosis_report.append("微循环：正常（平均微循环：{:.2f}）。".format(avg_bk))

    # 疲劳指数诊断
    if avg_fatigue > 25:
        diagnosis_report.append("疲劳状态：正常（平均疲劳状态：{:.2f}）。".format(avg_fatigue))
    elif 15 < avg_fatigue <= 25:
        diagnosis_report.append(
            "疲劳状态：较高（平均疲劳状态：{:.2f}）。轻度疲劳，表现为神疲乏力、注意力不集中、专注力差或者精神焦虑、紧张等。".format(
                avg_fatigue))
    else:
        diagnosis_report.append(
            "疲劳状态：过高（平均疲劳状态：{:.2f}）。过度疲劳，表现为周身乏力、记忆力减退、情绪波动较大以及睡眠质量较差等。".format(
                avg_fatigue))

    return "\n".join(diagnosis_report)


class FingerDataThread(QThread):
    data_received = pyqtSignal(str)  # 定义信号，发送诊断报告

    def __init__(self, serial_port='COM4', baudrate=38400, parent=None):
        super().__init__(parent)
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.running = True

    def run(self):
        report = finger_detect(self.serial_port, self.baudrate, packet_limit=30)  # 调用封装好的函数
        self.data_received.emit(report)  # 发送诊断报告
        print(report)
    def stop(self):
        self.running = False
        self.wait()