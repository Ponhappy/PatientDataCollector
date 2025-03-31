# 创建一个线程类来处理聊天请求
from PyQt5.QtCore import QThread, pyqtSignal

class ChatThread(QThread):
    response_ready = pyqtSignal(str)
    
    def __init__(self, chat_model, message):
        super().__init__()
        self.chat_model = chat_model
        self.message = message
        
    def run(self):
        try:
            response = self.chat_model.get_answer(self.message)
            self.response_ready.emit(response)
        except Exception as e:
            self.response_ready.emit(f"发生错误: {str(e)}")