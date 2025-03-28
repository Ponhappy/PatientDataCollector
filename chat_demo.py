from chat_model.cloud_chat import CloudChat
from chat_model.local_chat import LocalChat
import json
'''
请注意，不要挂梯子，不然会连接失败
'''

if __name__=="__main__":
    mode=1

    # #1、使用云端大模型
    if mode==1:
        # 必填项
        api_key="sk-d69f89a753d74b399a9404194d611aaa"
        base_url="https://api.deepseek.com"
        # 非必填 可使用默认参数
        prompt="你是一个中医大神"
        ds_v3='deepseek-chat'
        ds_r1='deepseek-reasoner'
        model=ds_v3#因为v3便宜些
        history_file="chat_model/cloud_history.json"

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


    # #2、利用ollama使用本地大模型
    elif mode==2:
        # 不准备用了 有点麻烦 直接采用1和3的方法
        print("通过ollama使用本地大模型")
        # LocalModel=LocalChat()
        # think_text, answer_text = LocalModel.get_answer(question)
        # print("*********************以下是本地大模型回答*******************")
        # print("Think 部分：", think_text)
        # print("Answer 部分：", answer_text)

    # #3、直接使用本地大模型（待加），预计使用ds-vl，蹭一下多模态
    elif mode==3:
        print("直接使用本地大模型")
