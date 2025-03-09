from openai import OpenAI
'''
请注意，不要挂梯子，不然会连接失败
'''
class CloudChat():
    def __init__(self,api_key,base_url,prompt="",model="deepseek-chat"):
        self.model=model
        self.api_key=api_key
        self.base_url=base_url
        self.prompt=prompt
    
    def get_answer(self,question):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content":self.prompt},
                {"role": "user", "content":question},
            ],
            stream=False
        )
        answer=response.choices[0].message.content
        return answer


if __name__ == "__main__":
    question = "What's your name?"
    api_key="sk-d69f89a753d74b399a9404194d611aaa"
    base_url="https://api.deepseek.com"

    ###########################################
    #两种模型，R1贵一些，model默认为V3，prompt默认为空string型
    ds_v3='deepseek-chat'
    ds_r1='deepseek-reasoner'
    prompt="你是一个中医大神"
    ###########################################
    
    chat_model = CloudChat(api_key,base_url)
    answer_text = chat_model.get_answer(question)
    print("Answer 部分：", answer_text)