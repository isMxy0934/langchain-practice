from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
load_dotenv()

llm = init_chat_model(model="deepseek-chat",model_provider="deepseek")

class Joke(BaseModel):
    """Joke to tell user."""
    setup: str = Field(..., description="The setup of the joke")
    punchline: str = Field(..., description="The punchline of the joke")
    rating: int = Field(..., description="How funny the joke is, from 1 to 10")
    
    
structured_llm = llm.with_structured_output(Joke)
result = structured_llm.invoke("Tell me a joke about a cat.")

print(result)
