from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START,MessagesState,StateGraph
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,AIMessage
from typing_extensions import Annotated, TypedDict
from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("deepseek-chat", model_provider="deepseek")
config = {"configurable": {"thread_id": "abc123"}}


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

workflow = StateGraph(state_schema=State)

def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {'messages':response}

workflow.add_edge(START,'model')
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages, "language": "Chinese"}, config)
output["messages"][-1].pretty_print()  


query = "What is my name?"

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages, "language": "Chinese"}, config)
output["messages"][-1].pretty_print()