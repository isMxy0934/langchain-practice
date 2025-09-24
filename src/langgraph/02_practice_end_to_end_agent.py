from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

config = {"configurable": {"thread_id": "abc123"}}
memory = MemorySaver()
model = init_chat_model(model="deepseek-chat",model_provider="deepseek")
search = TavilySearch(max_results=2)
tools = [search]

agent_executor = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=memory
)

input_message = {
    "role": "user",
    "content": "Hi, I'm Bob and I live in SF.",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()


input_message = {
    "role": "user",
    "content": "What's the weather where I live?",
}

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()