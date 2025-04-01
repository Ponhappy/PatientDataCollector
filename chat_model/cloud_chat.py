from openai import OpenAI
import os 
import json
from datetime import datetime
from markdown import markdown  # 需要安装python-markdown包

'''
请注意，不要挂梯子，不然会连接失败（好像又可以挂梯子使用了，不太懂，估计跟地区有关）
'''
class CloudChat():
    def __init__(self, api_key="", base_url="https://api.deepseek.com/v1", model="deepseek-reasoner", system_prompt="", history_file="history.json"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.history_file = history_file
        self.messages = []
        
        # 初始化消息历史
        self.load_history()
    
    def load_history(self):
        """从文件加载历史记录"""
        self.messages = []
        
        # 添加系统提示
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        
        # 读取历史记录文件
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                
                # 收集诊断报告
                diagnostic_reports = []
                # 将历史记录转换为API所需格式，确保交替出现
                last_role = None
                user_messages_buffer = []
                
                for message in history:
                    role = message.get("role")
                    content = message.get("content", "")
                    
                    # 收集诊断报告，但不立即添加到消息列表
                    if role == "user" and content.startswith("[") and "报告]" in content:
                        diagnostic_reports.append(content)
                        continue
                    
                    # 如果是普通用户消息或助手消息
                    if role in ["user", "assistant"]:
                        # 如果当前角色与上一角色相同(都是用户消息)，则合并
                        if role == last_role == "user":
                            user_messages_buffer.append(content)
                        else:
                            # 如果有缓冲的用户消息，先合并并添加
                            if user_messages_buffer and role != "user":
                                self.messages.append({"role": "user", "content": "\n".join(user_messages_buffer)})
                                user_messages_buffer = []
                            # 添加当前消息
                            self.messages.append({"role": role, "content": content})
                            last_role = role
                
                # 处理剩余缓冲的用户消息
                if user_messages_buffer:
                    self.messages.append({"role": "user", "content": "\n".join(user_messages_buffer)})
                
                # 处理诊断报告
                if diagnostic_reports:
                    # 如果最后一条消息是助手的，可以安全添加诊断报告作为新的用户消息
                    if self.messages and self.messages[-1]["role"] == "assistant":
                        self.messages.append({"role": "user", "content": "\n\n".join(diagnostic_reports)})
                    # 如果最后是用户消息，合并到现有消息中
                    elif self.messages and self.messages[-1]["role"] == "user":
                        self.messages[-1]["content"] += "\n\n" + "\n\n".join(diagnostic_reports)
                    # 如果没有消息历史，直接添加诊断报告作为第一条用户消息
                    else:
                        self.messages.append({"role": "user", "content": "\n\n".join(diagnostic_reports)})
                
                # 确保第一条消息是用户消息（系统消息之后）
                if len(self.messages) > 0 and self.messages[0]["role"] == "system":
                    if len(self.messages) > 1 and self.messages[1]["role"] != "user":
                        # 插入空的用户消息
                        self.messages.insert(1, {"role": "user", "content": "请根据以下诊断数据进行综合分析"})
                elif len(self.messages) > 0 and self.messages[0]["role"] != "user":
                    # 在开头插入用户消息
                    self.messages.insert(0, {"role": "user", "content": "请根据以下诊断数据进行综合分析"})
                
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"加载历史记录出错: {str(e)}")
                # 如果历史记录无法加载，仍然确保有系统提示
                self.messages = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
    
    def save_history(self):
        """保存历史记录到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        
        # 读取现有的历史记录（包含时间戳和诊断报告）
        full_history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    full_history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                full_history = []
        
        # 找到最新添加的消息
        if len(self.messages) > 0:
            last_message = self.messages[-1]
            if last_message["role"] == "assistant":
                # 添加时间戳
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_entry = {
                    "role": "assistant",
                    "content": last_message["content"],
                    "timestamp": timestamp
                }
                full_history.append(new_entry)
        
        # 保存更新后的历史记录
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(full_history, ensure_ascii=False, indent=2, fp=f)
    
    def get_answer(self, question):
        """获取AI回答"""
        # 检查消息历史中最后一条消息的角色
        if self.messages and self.messages[-1]["role"] == "user":
            # 如果最后一条是用户消息，先添加一个空的助手回复，确保交替顺序
            self.messages.append({"role": "assistant", "content": "我了解了，请继续。"})
            
            # 读取现有历史记录，添加这个虚拟助手消息
            full_history = []
            if os.path.exists(self.history_file):
                try:
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        full_history = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    full_history = []
            
            # 添加虚拟助手消息到完整历史
            virtual_assist = {
                "role": "assistant",
                "content": "我了解了，请继续。",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "virtual": True  # 标记为虚拟消息，UI可能需要特殊处理
            }
            full_history.append(virtual_assist)
            
            # 保存更新的历史记录
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(full_history, ensure_ascii=False, indent=2, fp=f)
        
        # 添加用户问题到消息历史
        self.messages.append({"role": "user", "content": question})
        
        # 记录用户消息到完整历史记录
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 读取现有历史记录
        full_history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    full_history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                full_history = []
        
        # 添加用户消息
        user_entry = {
            "role": "user",
            "content": question,
            "timestamp": timestamp
        }
        full_history.append(user_entry)
        
        # 保存更新的历史记录
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(full_history, ensure_ascii=False, indent=2, fp=f)
        
        # 调用API获取回答
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=False
        )
        
        # 根据模型类型处理返回结果
        if self.model == "deepseek-chat":
            answer = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": answer})
            self.save_history()
            return answer  # 只返回答案
        else:
            think = getattr(response.choices[0].message, 'reasoning_content', '')
            answer = response.choices[0].message.content
            full_response = f"{think}|||{answer}"
            self.messages.append({"role": "assistant", "content": full_response})
            self.save_history()
            
            try:
                # 转换Markdown为HTML
                formatted_response = markdown(
                    full_response,
                    extensions=['fenced_code', 'tables', 'nl2br']
                )
                
                # 添加CSS样式
                formatted_response = f"""
                <html>
                <head>
                <style>
                    body {{ 
                        font-family: 'Microsoft YaHei', SimSun, sans-serif;
                        font-size: 14px;
                        line-height: 1.6;
                        color: #333;
                    }}
                    strong {{ color: #2c3e50; }}
                    em {{ color: #27ae60; }}
                    code {{
                        background: #f8f9fa;
                        padding: 2px 4px;
                        border-radius: 3px;
                        font-family: Consolas, Monaco, monospace;
                    }}
                    pre {{
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 6px;
                        overflow-x: auto;
                    }}
                    table {{
                        border-collapse: collapse;
                        margin: 15px 0;
                        width: 100%;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #3498db;
                        color: white;
                    }}
                </style>
                </head>
                <body>
                {formatted_response}
                </body>
                </html>
                """
                
                self.messages.append({"role": "assistant", "content": formatted_response})
                return formatted_response
                
            except Exception as e:
                print(f"获取回答错误: {str(e)}")
                return "生成回答时发生错误"


if __name__ == "__main__":
    # 必填项
    api_key="sk-d69f89a753d74b399a9404194d611aaa"
    base_url="https://api.deepseek.com"
    # 非必填 可使用默认参数
    prompt="你是一个中医大神"
    ds_v3='deepseek-chat'
    ds_r1='deepseek-reasoner'
    # model=ds_v3#因为v3便宜些
    model =ds_r1
    history_file="history.json"

    chat_model = CloudChat(api_key,base_url,prompt,model,history_file)

    while True:
        question=input("问题：")
        if question.lower() in ["exit", "quit"]:
            print("对话结束")
            break
        answer = chat_model.get_answer(question)
        print("回答：",answer)

    history_messages=[]
    with open(history_file,'r',encoding='utf-8')as f:
        history_messages=json.load(f)
    print("历史消息如下：")
    print(history_messages)
