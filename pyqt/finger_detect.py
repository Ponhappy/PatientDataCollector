# finger_detect.py

import serial
import datetime
import openpyxl

def save_finger_pulse(serial_port, file_path):
    # 设置串口参数
    port = serial_port  # 动态传递串口端口
    baudrate = 115200
    timeout = 1
    # 初始化串口
    ser = serial.Serial(port, baudrate, timeout=timeout)
    print("串口无问题")

    # 创建一个新的 Excel 工作簿和工作表
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sensor Data"

    # 添加表头
    ws.append(["Timestamp", "Waveform", "Heartbeat", "Heart Rate", "HRV"])

    try:
        row_count = 0  # 用于记录已写入的行数
        while True:
            flag = ser.readline()
            if flag and flag[-1] == 0xff:  # 检查最后一个字节是否为 0xff
                print("无效输入")
                break
            # 读取一行数据
            data = ser.readline().decode('ascii').strip()

            # 检查数据格式，确保包含四个由逗号分隔的值
            if data:
                parts = data.split(',')
                if len(parts) == 4:
                    # 获取当前时间戳
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                    # 将数据和时间戳一起写入 Excel 文件
                    try:
                        ws.append([
                            timestamp,
                            float(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3])
                        ])
                        row_count += 1
                    except ValueError:
                        print(f"数据格式错误: {data}")

                    # 每 1000 条记录保存一次文件以避免内存过多
                    if row_count % 1000 == 0:
                        wb.save(file_path)
                        print(f"已保存 {row_count} 条记录到 {file_path}")
    except KeyboardInterrupt:
        print("程序中断")
    finally:
        # 最后保存一次文件
        wb.save(file_path)
        print(f"Pulse data saved to {file_path}")

        # 关闭串口
        ser.close()
