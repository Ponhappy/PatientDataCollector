import subprocess
import time
import requests
from ollama import chat
'''
请注意，不要挂梯子，不然会连接失败
'''
class LocalChat():
    def __init__(self, prompt="", model='deepseek-r1:1.5b'):
        self.prompt = prompt
        self.model = model

        if not self.is_ollama_running():
            self.start_ollama()

    def is_ollama_running(self):
        """检查 Ollama 服务器是否在运行"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_for_ollama(self, timeout=30):
        """等待 Ollama 服务器启动"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_ollama_running():
                print("✅ Ollama 服务器已成功启动！")
                return True
            time.sleep(2)
        print("❌ Ollama 启动失败，请检查手动运行 `ollama serve`")
        return False

    def test_ollama_ready(self):
        """测试 Ollama 是否能正常回答"""
        try:
            response = chat(model=self.model, messages=[{"role": "user", "content": "测试"}])
            return "message" in response
        except Exception:
            return False

    def start_ollama(self):
        """启动 Ollama 服务器"""
        print("Ollama 服务器未运行，正在启动...")
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if self.wait_for_ollama():
            time.sleep(3)  # 再等待几秒，确保模型加载
            if self.test_ollama_ready():
                print("🎉 Ollama 服务器已准备就绪！")
            else:
                print("⚠️ Ollama 服务器已启动，但仍无法响应对话")
        else:
            print("❌ Ollama 启动失败！")


    def get_answer(self, question):
        if self.is_ollama_running():
            print("Ollama 服务器已启动，开始对话...")
            try:
                response = chat(model=self.model, messages=[{'role': 'user', 'content': question}])
                text = response['message']['content']

                if "</think>" not in text:
                    return None, text

                parts = text.split("</think>")
                think_text = parts[0].replace("<think>", "").strip()
                answer_text = parts[1].strip()
                return think_text, answer_text
            except Exception as e:
                print(f"❌ 请求 Ollama 失败: {e}")
                return None, None
        else:
            print("❌ Ollama 未能启动，请手动检查")
            return None, None

if __name__ == "__main__":
    question = "What's your name?"
    
    ###########################################
    #两种模型，model默认为以下，prompt默认为空string型
    #此时采用本地模型，因此要保证本地有提前下载才能使用
    model='deepseek-r1:1.5b'
    prompt="你是一个中医大神"
    ###########################################

    chat_model = LocalChat()
    think_text, answer_text = chat_model.get_answer(question)
    print("Think 部分：", think_text)
    print("Answer 部分：", answer_text)
