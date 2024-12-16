import serial
import csv
import time
import sys
from datetime import datetime
import os

# 从命令行参数获取保存文件路径
if len(sys.argv) < 2:
    print("未提供文件路径，程序退出！")
    sys.exit(1)

csv_filename = sys.argv[1]

# 确保文件路径存在
file_dir = os.path.dirname(csv_filename)
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

# 打开串口连接
ser = serial.Serial('COM3', baudrate=38400, timeout=1, bytesize=8, stopbits=1, parity='N')

# 确保串口打开
if not ser.is_open:
    ser.open()

# 记录开始时间
start_time = time.time()

# 存储接收到的脉搏数据
received_data = []

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
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Data"])  # 写入表头

        while True:
            if ser.in_waiting > 0:  # 检查串口是否有数据
                # 读取一行数据
                raw_data = ser.read(ser.in_waiting)  # 读取所有可用的数据
                hex_data = raw_data.hex().upper()  # 转换为大写的十六进制字符串

                # 将十六进制数据拆分成字节，并用空格分隔
                formatted_data = " ".join([hex_data[i:i + 2] for i in range(0, len(hex_data), 2)])

                # 时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                # 保存数据（时间戳，十六进制字节数据）
                received_data.append([timestamp, formatted_data])
                writer.writerow([timestamp, formatted_data])  # 写入数据行

                print(formatted_data)

            # 你可以根据需要停止或限定读取时长
            if len(received_data) >= 30:
                break

            # 为了避免高CPU占用，可以适当休息一下
            time.sleep(0.1)

except KeyboardInterrupt:
    print("程序被中断，保存数据...")
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for data in received_data:
            writer.writerow(data)

finally:
    # 关闭串口连接
    ser.close()
    print("串口连接已关闭。")