from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file


llm = init_chat_model(model="deepseek-chat",model_provider="deepseek",temperature=0)

print(llm.invoke("Hello, how are you?"))