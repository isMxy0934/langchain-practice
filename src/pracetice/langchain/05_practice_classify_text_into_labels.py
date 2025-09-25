from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model(model="deepseek-chat",model_provider="deepseek")

prompt_template = ChatPromptTemplate.from_template(
  """
  Extract the desired information from the following passage.

  Only extract the properties mentioned in the 'Classification' function.

  Passage:
  {input}
  """
)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text",enum=["happy","neutral","sad"])
    aggressiveness: str = Field(description="How aggressive the text is on a scale from 1 to 10",enum=[1,2,3,4,5])
    language: str = Field(description="The language of text is written in",enum=["English","Chinese","Spanish","French","German","Japanese","Korean"])

structured_llm = llm.with_structured_output(Classification)

# input = "I hate this product! It is the worst thing I have ever bought. I will never buy it again."
input = "我讨厌这个产品！这是我买过的最糟糕的东西。我再也不会买了。"
prompt = prompt_template.invoke({"input": input})

print (f"Prompt:{prompt}\n")

response = structured_llm.invoke(prompt)
print (f"Response:{response.model_dump()}\n")