# wrist_detect.py

import serial
import csv
import time
from datetime import datetime

def save_wrist_pulse(serial_port, csv_filename):
    # 打开串口连接
    ser = serial.Serial(serial_port, baudrate=38400, timeout=1, bytesize=8, stopbits=1, parity='N')

    # 确保串口打开
    if not ser.is_open:
        ser.open()

    # 向串口发送启动命令 (0x8A)
    ser.write(bytes([0x8A]))  # 发送单个字节 0x8A 启动设备
    time.sleep(0.5)  # 等待设备响应的时间

    # 记录开始时间
    start_time = time.time()

    # 存储接收到的脉搏数据
    received_data = []
    timestamps = []  # 存储时间戳

    # 解析数据包的函数
    def parse_packet(packet):
        # 校验包头是否正确
        if packet[:2] != b'\xaa\x55':
            return None

        token = packet[2]
        length = packet[3]

        # 校验令牌是否有效
        if token not in [255, 80, 81, 82, 83]:
            return None

        # 解析类型
        packet_type = packet[4]
        if token == 82 and packet_type == 1:
            # 提取数据（去掉包头、令牌、长度、类型，最后一位为校验和）
            data = packet[5:-1]
            int_data = list(data)
            pulse_flag = (data[0] >> 7) & 1  # bit7为脉搏搏动标志

            waveform = int_data  # 波形数据从第1个字节开始
            # 波形数据范围是 0-127
            for ff in range(len(waveform)):
                if waveform[ff] > 127:
                    waveform[ff] = 127

            return pulse_flag, waveform
        return None

    # 读取数据并保存为 CSV 文件
    try:
        while True:
            if ser.in_waiting > 0:  # 检查串口是否有数据
                # 读取所有可用的数据
                raw_data = ser.read(ser.in_waiting)
                # 解析数据包
                parsed = parse_packet(raw_data)
                if parsed:
                    pulse_flag, waveform = parsed
                    # 时间戳
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    # 保存数据（时间戳，脉搏标志，波形数据）
                    received_data.append([timestamp, pulse_flag, waveform])
                    print(f"Timestamp: {timestamp}, Pulse Flag: {pulse_flag}, Waveform: {waveform}")
            # 假设读取 50 条数据，然后结束循环
            if len(received_data) >= 50:
                break

            # 为了避免高CPU占用，可以适当休息一下
            time.sleep(0.1)

    finally:
        # 关闭串口连接
        ser.close()
        print("Serial connection closed.")

    # 将数据保存为 CSV 文件
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Pulse Flag", "Waveform"])  # 写入表头
        for data in received_data:
            writer.writerow(data)  # 写入每一行数据

    print(f"Pulse data saved to {csv_filename}")
