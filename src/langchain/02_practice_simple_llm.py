from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(model="deepseek-chat", model_provider="deepseek")

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to Chinese."),
    HumanMessage(content="'Hello, world!'")
]

print(model.invoke(messages))


for token in model.stream(messages):
    print(token, end="|")