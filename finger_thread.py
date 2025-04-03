import serial
import struct
import json
# from threading import Thread
from datetime import datetime
from PyQt5.QtCore import QThread, pyqtSignal
import os


class FingerDataThread(QThread):
    report_received = pyqtSignal(str)  # 发送 HRV 诊断报告
    wave_received = pyqtSignal(object)  # 传递波形数据
    def __init__(self, serial_port="COM9", baudrate=38400, duration=30):
        super().__init__()
        
        # 接收串口通信的参数
        self.ser=None
        if isinstance(serial_port, int):
            self.serial_port = str(serial_port)  # 将传入的整数端口转换为字符串
        else:
            self.serial_port = serial_port
        self.duration=duration
        self.baudrate = baudrate
        self.running = True
        
        # 数据存储的位置
        self.dir=''

        # 不同数据的保存
        self.buffer = bytearray()
        self.packet_data=[]
        self.parameter_data=[]
        self.waveform_data=[]

    def start_serial(self):
        if self.ser ==None:
            self.ser = serial.Serial(self.serial_port, self.baudrate)
            return True
        else:
            return True


    def run(self):
        if not self.start_serial():# 未打开串口通信
            return
        else:
            while self.running:
                if self.ser.in_waiting:
                    self.buffer.extend(self.ser.read(self.ser.in_waiting))
                    index = 0
                    # 为了保证至少有index+7个字节在buffer中
                    while index + 7 <= len(self.buffer):
                        if self.buffer[index] == 0xaa and self.buffer[index + 1] == 0x55:
                            length = self.buffer[index + 3]# 长度之后，类型+数据+校验和的总长
                            total_length = length + 4  # 加上帧头、令牌、长度的长度
                            if index + total_length <= len(self.buffer):
                                packet = self.buffer[index:index + total_length]
                                token = packet[2]
                                type_byte = packet[4]
                                content = packet[5:-1]
                                # 存储正确数据包
                                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                                self.save_data(timestamp,'packet',packet.hex())
                                if type_byte == 0x01 and token == 0x53:  # 主动上传参数数据包
                                    self.handle_parameter_data(content,timestamp)
                                elif type_byte in [0x01, 0x02] and token == 0x52:  # 主动上传波形数据包
                                    self.handle_waveform_data(type_byte, content,timestamp)
                                else:# 别的情况
                                    pass
                                index += total_length
                            else:
                                break
                        else:
                            index += 1
                    # 移除已经处理过的数据
                    del self.buffer[:index]

    
    
    def stop(self):
        self.save_to_json()
        self.running = False
        self.ser.close()
        report=self.get_report()
        self.report_received.emit(report)  # 发送诊断报告
        self.wait()


    def save_to_json(self):
        packet_fp=os.path.join(self.dir,'packet.json')
        parameter_fp=os.path.join(self.dir,'parameter.json')
        waveform_fp=os.path.join(self.dir,'waveform.json')
        with open(packet_fp, 'w') as f:
            json.dump(self.packet_data, f, indent=4)
        with open(parameter_fp, 'w') as f:
            json.dump(self.parameter_data, f, indent=4)
        with open(waveform_fp, 'w') as f:
            json.dump(self.waveform_data, f, indent=4)
    

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
            self.wave_received.emit(data)
        else:# 其他类型数据
            pass


    def get_report(self):
        para_len=len(self.parameter_data)
        report=''
        spo2_re=''
        pr_re=''
        pi_re=''
        
        ave_SpO2=0
        ave_PR=0
        ave_PI=0
        cnt=0

        for i in range(para_len):
            if self.parameter_data[i]['data']['spo2']!=0:
                ave_SpO2+=self.parameter_data[i]['data']['spo2']
                ave_PR+=self.parameter_data[i]['data']['pr']
                ave_PI+=self.parameter_data[i]['data']['pi']
                cnt+=1
                
        if cnt==0:#避免出现除数为0
            cnt+=1
        ave_SpO2/=cnt
        ave_PR/=cnt
        ave_PI/=cnt
        # SPO2
        if ave_SpO2>100:
            spo2_re=(
                f'平均血氧饱和度已高达【{ave_SpO2:.2f}%】，一般无特殊临床表现，因为人体血氧饱和度在正常生理状态下很难超过 100%。'
                '如果测量值显示过高，可能是测量仪器出现故障或测量方法不正确。'
                '但如果排除测量问题，真正出现血氧过高，可能提示吸氧浓度过高或氧疗不当，'
                '长期可能会导致氧中毒，对肺部和其他器官造成损伤。'
            )
        elif ave_SpO2<90:
            spo2_re=(
                f'平均血氧饱和度已下降到【{ave_SpO2:.2f}%】，'
                '呼吸频率加快，呼吸深度加深，可能伴有喘息、胸闷等症状。患者会感到头晕、乏力、心慌，严重时可能出现意识模糊、昏迷等。'
                '长期或严重的低血氧饱和度会导致组织和器官缺氧，引起多器官功能损害，如心脏、大脑、肝脏、肾脏等功能障碍。'
                '可能引发心力衰竭、呼吸衰竭、脑损伤等严重疾病，甚至危及生命。'
            )
        else:
            spo2_re=(
                f'平均血氧饱和度为【{ave_SpO2:.2f}%】，身体各组织和器官能够获得充足的氧气供应，维持正常的生理功能。'
                '身体处于良好的氧合状态，能够保证细胞的正常代谢和功能，'
                '有助于维持身体的健康和正常活动，降低因缺氧导致的各种疾病风险。'
            )
        # PR
        if ave_PR>100:
            pr_re=(
                f'平均脉率已高达【{ave_PR:.2f}bpm】，会感觉心慌、心跳剧烈，可能伴有呼吸急促、头晕等症状。'
                '在运动或情绪激动后，脉率升高是正常的生理反应，但如果在静息状态下持续心动过速，可能会出现乏力、疲劳等不适。'
                '严重时，可能会出现心绞痛、心力衰竭等症状。'
                '长期的心动过速会增加心脏的负担，导致心肌肥厚，甚至发展为心肌病。'
                '同时，也会增加心律失常、中风和心脏骤停等心血管疾病的发生风险，对身体健康造成严重威胁。'
            )
        elif ave_PR<60:
            pr_re=(
                f'平均脉率已下降到【{ave_PR:.2f}bpm】，可能会出现头晕、乏力、眼前发黑等症状，严重时可能导致晕厥。'
                '在运动或体力活动时，会感到气短、心慌，身体耐力下降。'
                '如果是由于心脏传导系统疾病等原因引起的心动过缓，还可能伴有胸闷、胸痛等症状。'
                '脉率过低会导致心脏输出量减少，影响全身的血液供应，尤其是大脑、心脏等重要器官。'
                '长期心动过缓可能引发心力衰竭、脑供血不足等并发症，增加心血管疾病的风险，影响生活质量和身体健康。'
            )
        else:
            pr_re=(
                f'平均脉率为【{ave_PR:.2f}bpm】，心脏有规律地收缩和舒张，脉搏跳动有力且节律整齐。'
                '身体能够保持正常的血液循环，为各组织器官提供充足的血液灌注。'
                '在日常活动中，身体能够适应不同的运动强度和生理需求，不会出现心慌、胸闷等不适症状。'
                '正常的脉率表明心脏功能和血液循环系统处于良好状态，能够维持身体的正常代谢和功能活动，有助于预防心血管疾病等健康问题。'
            )
        # PI
        if ave_PI/10>20:
            pi_re=(
                f'平均灌注指数已高达【{ave_PI/10:.2f}%】，可能表示外周血管扩张，血流速度加快。'
                '皮肤可能会出现潮红、发热的现象，尤其是在测量部位附近。'
                '一般情况下，单纯的 PI 过高可能不会有明显的不适症状，'
                '但如果是由于某些疾病引起的，如甲状腺功能亢进、严重感染等，可能会伴有相应疾病的其他症状，如多汗、心慌、发热等。'
                '同时，也可能提示身体存在某些疾病状态，需要进一步检查和治疗，以避免病情恶化。'
            )
        elif ave_PI/10<1:
            pi_re=(
                f'平均灌注指数已下降到【{ave_PI/10:.2f}%】，可能提示外周血管收缩或血容量不足，皮肤可能会出现发凉、苍白等表现，尤其是肢体末端。'
                '灌注指数过低会导致局部组织血液灌注不足，引起组织缺氧和代谢紊乱。'
                '长期可能导致组织坏死、溃疡等并发症，还可能影响伤口愈合，增加感染的风险。'
                '如果是全身性的灌注不足，会影响多个器官的功能，严重时可导致休克。'
            )
        else:
            pi_re=(
                f'平均灌注指为【{ave_PI/10:.2f}%】，表示外周血管血流灌注良好。'
                '手指、脚趾等部位的皮肤温暖、红润，毛细血管充盈时间正常。'
                '身体各组织能够得到充足的血液供应，以维持正常的代谢和功能。'
            )
        report=f'{spo2_re}\n\n{pr_re}\n\n{pi_re}'
        
        return report

    def handle_parameter_data(self, content,timestamp):
        # print('*********************查询参数数据***********************')
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
        # print(f"血氧饱和度(SpO2): {spo2}%")
        # print(f"脉率(PR): {pr} bpm")
        # print(f"灌注指数(PI): {pi / 1000}")
        # print(f"探头断开状态: {'是' if probe_disconnected else '否'}")
        # print(f"探头脱落状态: {'是' if probe_off else '否'}")
        # print(f"脉搏搜索状态: {'是' if pulse_searching else '否'}")
        # print(f"检查探头状态: {'是' if check_probe else '否'}")
        # print(f"运动检测状态: {'是' if motion_detected else '否'}")
        # print(f"低灌注状态: {'是' if low_perfusion else '否'}")
        # print(f"工作模式: {mode_mapping[mode]}")
        parameter_data = {
            'spo2': spo2,
            'pr': pr,
            'pi': pi,
            'status_byte': status_byte
        }
        self.save_data(timestamp, 'parameter', parameter_data)

    def handle_waveform_data(self, type_byte, content,timestamp):
        # print('*********************查询波形数据***********************')
        waveform_data = []
        if type_byte == 0x01:
            for i in range(0, len(content), 1):
                pulse_flag = (content[i] >> 7) & 0x1
                waveform_value = content[i] & 0x7f
                waveform_data.append((pulse_flag, waveform_value))
            # print(f"归一化波形数据: {waveform_data}")
        
        elif type_byte == 0x02:
            ir_data = []
            red_data = []
            for i in range(0, len(content), 8):
                ir_sample = struct.unpack('<I', content[i:i + 4])[0]
                red_sample = struct.unpack('<I', content[i + 4:i + 8])[0]
                ir_data.append(ir_sample)
                red_data.append(red_sample)
            # print(f"未归一化红外波形数据: {ir_data}")
            # print(f"未归一化红光波形数据: {red_data}")
            waveform_data={'ir_data': ir_data, 'red_data': red_data}
        self.save_data(timestamp, 'waveform', waveform_data)