from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model(model="deepseek-chat", model_provider="deepseek")

system_template = "You are a helpful assistant that translates English to {language}."

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",system_template),
        ("user","{text}"),
    ]
)


prompt = prompt_template.invoke({"language":"Chinese","text":"I love programming."})


print("Prompt:",prompt.to_messages())
print("===")
response = model.invoke(prompt)
print("Response:",response)