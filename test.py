# from hugchat import hugchat
# chatbot = hugchat.ChatBot(cookie_path="cookies.json")  # or cookies=[...]
# print("Chatbot initialized")
# prompt = "Hi this is a test"
# print("Calling HuggingFace API with prompt: ", prompt)
# response = chatbot.chat(
#     text=prompt,
#     temperature=0.8)
# print("Raw response: ", response)

from bardapi import Bard

token = 'Wwg5tjJhZP8TwDzkNAh0bx75YwFGD1aLOuIgtuRn5DmQi_n5v-QQ3nDK3oNdQ9_pGDbReg.'

proxies = {
    # 'http': 'http://127.0.0.1:7890',
    # 'https': 'https://127.0.0.1:7890',
}
bard = Bard(token=token, timeout=30, proxies=proxies)
bard.get_answer("what is that")