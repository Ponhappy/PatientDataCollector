import serial
import struct
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DataReceiver:
    def __init__(self, port, baudrate=38400):
        self.ser = serial.Serial(port, baudrate)
        # 这一坨crc暂时没啥用
        self.crc_table = [
            0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83, 0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f, 0x41,
            0x9d, 0xc3, 0x21, 0x7f, 0xf8, 0xa2, 0x40, 0x1e, 0x5b, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82, 0xd0,
            0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0, 0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde, 0x3c, 0x62,
            0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d, 0x7c, 0x22, 0xc0, 0x9e, 0x1d, 0x43, 0xa1, 0xf9,
            0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5, 0x84, 0xda, 0x38, 0x66, 0xe5, 0xb6, 0x59, 0x07,
            0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58, 0x19, 0x47, 0xa5, 0xf9, 0x78, 0x26, 0xc4, 0x9a,
            0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6, 0xa7, 0x99, 0x1b, 0x45, 0xc6, 0x98, 0x7a, 0x24,
            0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b, 0x3a, 0x64, 0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
            0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f, 0x4e, 0x10, 0x92, 0xac, 0x2f, 0x71, 0x93, 0xcd,
            0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92, 0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec, 0x0e, 0x50,
            0xaf, 0xf1, 0x13, 0x4d, 0xc0, 0x90, 0x72, 0x2c, 0x6d, 0x33, 0xf1, 0x8f, 0x0c, 0x52, 0xb0, 0xee,
            0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1, 0xf0, 0xae, 0x4c, 0x12, 0x91, 0xcf, 0x2d, 0x73,
            0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49, 0x08, 0x56, 0xb4, 0xea, 0x69, 0x37, 0xd5, 0x8b,
            0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4, 0x95, 0xcb, 0x29, 0x77, 0xf4, 0xaa, 0x48, 0x16,
            0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a, 0x25, 0x75, 0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
            0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7, 0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b, 0x35
        ]
        self.buffer = bytearray()
        self.packet_data=[]
        self.parameter_data=[]
        self.waveform_data=[]

        # 以下的数据可以注释，这是为了方便实时看波形的
        self.cnt=0
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.x_data, self.y_data = [], []
        self.line, = self.ax.plot([], [], lw=2)
        self.start_time = None
        self.time_window = 10  # 时间窗口为10秒
        self.ax.set_xlim(0, self.time_window)
        self.ax.set_ylim(0, 150) 
        plt.yticks([])

    # 暂时没啥用的计算校验和
    def calculate_crc8(self, data):
        crc = 0
        for byte in data:
            crc = self.crc_table[crc ^ byte]
        return crc
    # 暂时没啥用，因为计算校验和有问题
    def send_query(self, token, type_byte, content=b''):
        head = b'\xaa\x55'
        length = len(content) + 2  # 数据内容长度加上类型和校验和的长度
        packet = head + bytes([token]) + bytes([length]) + bytes([type_byte]) + content
        crc = self.calculate_crc8(packet)
        packet += bytes([crc])
        self.ser.write(packet)

    def receive_and_parse(self):
        while True:
            if self.ser.in_waiting:
                self.buffer.extend(self.ser.read(self.ser.in_waiting))
                index = 0
                # 为了保证至少有index+7个字节在buffer中
                while index + 7 <= len(self.buffer):
                    if self.buffer[index] == 0xaa and self.buffer[index + 1] == 0x55:
                        # print("检测到数据开头head")
                        length = self.buffer[index + 3]# 长度之后，类型+数据+校验和的总长
                        total_length = length + 4  # 加上帧头、令牌、长度的长度
                        
                        if index + total_length <= len(self.buffer):
                            # print("至少有一条完整数据")
                            packet = self.buffer[index:index + total_length]
                            # print(f'该条完整数据为：{packet}')
                            token = packet[2]
                            type_byte = packet[4]
                            content = packet[5:-1]
                            received_crc = packet[-1]
                            calculated_crc = self.calculate_crc8(packet[:-1])
                            print(f'接受校验和：{received_crc}\t计算校验和：{calculated_crc}')
                            # 注意这里校验和计算有问题 需要再看看
                            if True:#calculated_crc != received_crc:
                                # print("接受校验和与计算校验和相等，该数据为一条完整数据")
                                # 存储正确数据包
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                                self.save_data(timestamp,'packet',packet.hex())
                                # print(f'packet:{packet}\t十六进制：{packet.hex()}')
                                if type_byte == 0x01 and token == 0xff:  # 查询产品ID应答
                                    self.handle_product_id(content)
                                elif type_byte == 0x01 and token == 0x51:  # 查询版本信息应答
                                    self.handle_version_info(content)
                                elif type_byte == 0x02 and token == 0x51:  # 查询工作状态应答
                                    self.handle_working_status(content)
                                elif type_byte == 0x01 and token == 0x53:  # 主动上传参数数据包
                                    self.handle_parameter_data(content,timestamp)
                                elif type_byte in [0x01, 0x02] and token == 0x52:  # 主动上传波形数据包
                                    self.handle_waveform_data(type_byte, content,timestamp)
                            index += total_length
                        else:
                            break
                    else:
                        index += 1
                # 移除已经处理过的数据
                del self.buffer[:index]
    
    # 保存数据
    def save_data(self, timestamp, data_type, data):
        record = {
            'timestamp': timestamp,
            'data_type': data_type,
            'data': data
        }
        if data_type=='packet':
            self.packet_data.append(record)
        elif data_type=='parameter':
            self.parameter_data.append(record)
        elif data_type=='waveform':
            self.waveform_data.append(record)
            # 以下可注释，方便试试看波形
            if isinstance(data, list):
                if not self.start_time:
                    self.start_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                current_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                time_diff = (current_time - self.start_time).total_seconds()
                relative_time = time_diff % self.time_window
                cnt = time_diff // self.time_window
                
                # 清除超出时间窗口的所有旧数据
                if cnt>self.cnt:
                    self.cnt=cnt
                    self.y_data = []
                    self.x_data = []
                    
                for _, value in data:
                    self.x_data.append(relative_time)
                    self.y_data.append(value)
                # print(f"x长{len(self.x_data)}\ty长{len(self.y_data)}\tcnt为{cnt}")
                self.line.set_data(self.x_data, self.y_data)
                self.ax.relim()
                self.ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)
    # 保存到json文件
    def save_to_json(self, packet_fp='packet.json',parameter_fp='parameter.json',waveform_fp='waveform.json'):
        with open(packet_fp, 'w') as f:
            json.dump(self.packet_data, f, indent=4)
        with open(parameter_fp, 'w') as f:
            json.dump(self.parameter_data, f, indent=4)
        with open(waveform_fp, 'w') as f:
            json.dump(self.waveform_data, f, indent=4)

    def handle_product_id(self, content):
        print('*********************查询产品ID***********************')
        product_id = content.decode('ascii')
        print(f"产品ID: {product_id}")

    def handle_version_info(self, content):
        print('*********************查询产品版本***********************')
        software_version = content[0]
        hardware_version = content[1]
        print(f"软件版本: Ver{software_version >> 4}.{software_version & 0xf}")
        print(f"硬件版本: Ver{hardware_version >> 4}.{hardware_version & 0xf}")

    def handle_working_status(self, content):
        print('*********************查询工作状态***********************')
        status_byte = content[0]
        mode = (status_byte >> 6) & 0x3
        mode_mapping = {0: "成人模式", 1: "新生儿模式", 2: "动物模式", 3: "预留"}
        sending_status = (status_byte >> 5) & 0x1
        probe_status = (status_byte >> 4) & 0x1
        probe_off = (status_byte >> 3) & 0x1
        check_probe = (status_byte >> 2) & 0x1
        print(f"工作模式: {mode_mapping[mode]}")
        print(f"上行主动发送状态: {'允许' if sending_status else '禁止'}")
        print(f"探头连接状态: {'未连接' if probe_status else '已连接'}")
        print(f"探头脱落状态: {'是' if probe_off else '否'}")
        print(f"检查探头状态: {'是' if check_probe else '否'}")

    def handle_parameter_data(self, content,timestamp):
        print('*********************查询参数数据***********************')
        spo2 = content[0]
        pr_low = content[1]
        pr_high = content[2]
        pi = content[3]
        status_byte = content[4]
        pr = (pr_high << 8) + pr_low
        probe_disconnected = (status_byte >> 0) & 0x1
        probe_off = (status_byte >> 1) & 0x1
        pulse_searching = (status_byte >> 2) & 0x1
        check_probe = (status_byte >> 3) & 0x1
        motion_detected = (status_byte >> 4) & 0x1
        low_perfusion = (status_byte >> 5) & 0x1
        mode = (status_byte >> 6) & 0x3
        mode_mapping = {0: "成人模式", 1: "新生儿模式", 2: "动物模式", 3: "预留"}
        print(f"血氧饱和度(SpO2): {spo2}%")
        print(f"脉率(PR): {pr} bpm")
        print(f"灌注指数(PI): {pi / 1000}")
        print(f"探头断开状态: {'是' if probe_disconnected else '否'}")
        print(f"探头脱落状态: {'是' if probe_off else '否'}")
        print(f"脉搏搜索状态: {'是' if pulse_searching else '否'}")
        print(f"检查探头状态: {'是' if check_probe else '否'}")
        print(f"运动检测状态: {'是' if motion_detected else '否'}")
        print(f"低灌注状态: {'是' if low_perfusion else '否'}")
        print(f"工作模式: {mode_mapping[mode]}")
        parameter_data = {
            'spo2': spo2,
            'pr': pr,
            'pi': pi,
            'status_byte': status_byte
        }
        self.save_data(timestamp, 'parameter', parameter_data)


    def handle_waveform_data(self, type_byte, content,timestamp):
        print('*********************查询波形数据***********************')
        waveform_data = []
        if type_byte == 0x01:
            for i in range(0, len(content), 1):
                pulse_flag = (content[i] >> 7) & 0x1
                waveform_value = content[i] & 0x7f
                waveform_data.append((pulse_flag, waveform_value))
            print(f"归一化波形数据: {waveform_data}")
        
        elif type_byte == 0x02:
            ir_data = []
            red_data = []
            for i in range(0, len(content), 8):
                ir_sample = struct.unpack('<I', content[i:i + 4])[0]
                red_sample = struct.unpack('<I', content[i + 4:i + 8])[0]
                ir_data.append(ir_sample)
                red_data.append(red_sample)
            print(f"未归一化红外波形数据: {ir_data}")
            print(f"未归一化红光波形数据: {red_data}")
            waveform_data={'ir_data': ir_data, 'red_data': red_data}
        self.save_data(timestamp, 'waveform', waveform_data)


if __name__ == "__main__":
    port = 'COM9'  # 替换为实际的串口端口
    receiver = DataReceiver(port)

    # 服务器传送到设备的命令
    # # 查询产品ID
    # receiver.send_query(0xff, 0x01)
    # # 查询版本信息
    # receiver.send_query(0x51, 0x01)
    # # 查询工作状态
    # receiver.send_query(0x51, 0x02)
    
    
    # 接受设备传送到服务的回答
    try:
        ani = FuncAnimation(receiver.fig, lambda: None, interval=100)
        receiver.receive_and_parse()
    except KeyboardInterrupt:
        # 可以指定位置，不然就直接默认文件位置了
        receiver.save_to_json()
        plt.close()


    """
    # 校验和计算
    test_data = b'\xaa\x55\xff\x02\x01'
    expected_crc = 0x12  # 假设手动计算或其他工具得出的正确CRC值
    calculated_crc = receiver.calculate_crc8(test_data)
    if calculated_crc != expected_crc:
        print(f"CRC8计算错误，计算值: {hex(calculated_crc)}，期望值: {hex(expected_crc)}")
    """
    