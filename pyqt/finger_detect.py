import serial
import csv
import time
from datetime import datetime


def save_finger_pulse(csv_filename,COM):
    # 打开串口连接
    ser = serial.Serial(COM, baudrate=38400, timeout=1, bytesize=8, stopbits=1, parity='N')

    if not ser.is_open:
        print("串口没有打开")
        ser.open()
    print("串口已打开")
    # 向串口发送启动命令 (0x8A)
    ser.write(bytes([0x8A]))  # 发送单个字节 0x8A 启动设备
    time.sleep(0.5)  # 等待设备响应的时间

    # 存储接收到的脉搏数据
    received_data = []
    timestamps = []  # 存储时间戳



    # 读取数据并保存为 CSV 文件
    try:
        print("进入获取指尖数据")
        while True:
            # print("开始循环")
            # print("ser.in_waiting:",ser.in_waiting)
            if ser.in_waiting > 0:  # 检查串口是否有数据
                # 读取一行数据
                raw_data = ser.read(ser.in_waiting)  # 读取所有可用的数据
                print(raw_data)
                hex_data = raw_data.hex().upper()  # 转换为大写的十六进制字符串

                # 将十六进制数据拆分成字节，并用空格分隔
                formatted_data = " ".join([hex_data[i:i + 2] for i in range(0, len(hex_data), 2)])

                # 时间戳
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                # 保存数据（时间戳，十六进制字节数据）
                received_data.append([timestamp, formatted_data])
                timestamps.append(timestamp)
                # print("formatted_data:",formatted_data)

            # 你可以根据需要停止或限定读取时长
            # 假设你希望读取 100 行数据，然后结束循环
            if len(received_data) >= 10:
                
                print("达到10，退出循环")
                break

            # 为了避免高CPU占用，可以适当休息一下
            time.sleep(0.1)

    finally:
        print("关闭串口连接")
        # 关闭串口连接
        ser.close()
        print("Serial connection closed.")


    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp",  "Data"])  # 写入表头
        for idx, data in enumerate(received_data):
            writer.writerow([data[0],  data[1]])  # 写入每一行数据



# save_finger_pulse("1.csv","COM3")