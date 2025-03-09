from chat_model.cloud_chat import CloudChat
from chat_model.local_chat import LocalChat

'''
请注意，不要挂梯子，不然会连接失败
'''

if __name__=="__main__":
    question="你是谁？"
    #1、使用本地大模型
    LocalModel=LocalChat()
    think_text, answer_text = LocalModel.get_answer(question)
    print("*********************以下是本地大模型回答*******************")
    print("Think 部分：", think_text)
    print("Answer 部分：", answer_text)
    #2、使用云端大模型
    api_key="sk-d69f89a753d74b399a9404194d611aaa"
    base_url="https://api.deepseek.com"
    CloudModel=CloudChat(api_key,base_url)
    cloud_answer=CloudModel.get_answer(question)
    print("*********************以下是本地大模型回答*******************")
    print(cloud_answer)

