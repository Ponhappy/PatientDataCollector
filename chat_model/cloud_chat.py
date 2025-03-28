from openai import OpenAI
import os 
import json

'''
请注意，不要挂梯子，不然会连接失败（好像又可以挂梯子使用了，不太懂，估计跟地区有关）
'''
class CloudChat():
    def __init__(self,api_key,base_url,prompt="",model="deepseek-chat",history_file="history.json"):
        self.model=model
        self.api_key=api_key
        self.base_url=base_url
        self.prompt=prompt
        self.history_file=history_file
        self.messages=self.load_history()
        if not self.messages:
            self.messages = [{"role": "system", "content": prompt}]
    
    def save_history(self):
        with open(self.history_file,"w",encoding="utf-8")as f:
            json.dump(self.messages,f,ensure_ascii=False,indent=4)

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file,'r',encoding='utf-8')as f:
                return json.load(f)
        return []

    def get_answer(self,question):
        self.messages.append({"role": "user", "content": question})  # 记录用户问题
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stream=False
        )
        answer=response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": answer})  # 记录模型回复
        self.save_history()
        return answer


if __name__ == "__main__":
    # 必填项
    api_key="sk-d69f89a753d74b399a9404194d611aaa"
    base_url="https://api.deepseek.com"
    # 非必填 可使用默认参数
    prompt="你是一个中医大神"
    ds_v3='deepseek-chat'
    ds_r1='deepseek-reasoner'
    model=ds_v3#因为v3便宜些
    history_file="history.json"

    chat_model = CloudChat(api_key,base_url,prompt,model,history_file)

    while True:
        question=input("问题：")
        if question.lower() in ["exit", "quit"]:
            print("对话结束")
            break
        answer = chat_model.get_answer(question)
        print("ds回答：",answer)

    history_messages=[]
    with open(history_file,'r',encoding='utf-8')as f:
        history_messages=json.load(f)
    print("历史消息如下：")
    print(history_messages)
