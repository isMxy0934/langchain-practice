from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticToolsParser

load_dotenv()  # take environment variables from .env file


class add(BaseModel):
  """A tool for adding two numbers"""
  a: int = Field(..., description="The first number to add")
  b: int = Field(..., description="The second number to add")

class multiply(BaseModel):
  """A tool for multiplying two numbers"""
  a: int = Field(..., description="The first number to multiply")
  b: int = Field(..., description="The second number to multiply")


tools = [add, multiply]

llm = init_chat_model(model="deepseek-chat",model_provider="deepseek",temperature=0)

llm_with_tools = llm.bind_tools(tools)

print(llm_with_tools.invoke("Add 1 and 2").tool_calls)


parser = PydanticToolsParser(tools=tools)

chain = llm_with_tools | parser
result = chain.invoke("Multiply 3 and 4")
print(result)