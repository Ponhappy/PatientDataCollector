import serial
import datetime
import openpyxl


def save_finger_pulse(file_path):
    # 设置串口参数
    port = 'COM5'  # 替换为你的串口号
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
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    # 将数据和时间戳一起写入 Excel 文件
                    ws.append([timestamp, int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
                    row_count += 1

                    # 每 1000 条记录保存一次文件以避免内存过多
                    if row_count % 1000 == 0:
                        wb.save(file_path)

    except KeyboardInterrupt:
        print("程序中断")
    finally:
        # 最后保存一次文件
        wb.save(file_path)

        # 关闭串口
        ser.close()