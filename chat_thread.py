# 创建一个线程类来处理聊天请求
from PyQt5.QtCore import QThread, pyqtSignal
from datetime import datetime

class ChatThread(QThread):
    response_ready = pyqtSignal(tuple)  # 发送(响应内容, 时间戳)元组
    
    def __init__(self, chat_model, message, timestamp):
        super().__init__()
        self.chat_model = chat_model
        self.message = message
        self.timestamp = timestamp
        
    def run(self):
        try:
            response = self.chat_model.get_answer(self.message)
            response_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.response_ready.emit((response, response_timestamp))
        except Exception as e:
            self.response_ready.emit((f"发生错误: {str(e)}", self.timestamp))