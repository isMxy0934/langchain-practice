from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing_extensions import TypedDict,Annotated
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

config = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()
db = SQLDatabase.from_uri("sqlite:////Users/mu/Desktop/langchain-practice/db/Chinook.db")
llm = init_chat_model(model="deepseek-chat",model_provider="deepseek")

system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_message = "Question: {input}"


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
    
class QueryOutput(TypedDict):
    """Generated SQL query"""
    query: Annotated[str, ..., "Syntactically valid SQL query."]
    
    
query_prompt_template = ChatPromptTemplate.from_messages([
    ("system",system_message),
    ("user",user_message)
])    
    
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect":db.dialect,
            "top_k":10,
            "table_info":db.get_table_names(),
            "input":state["question"]
        }
    )
    
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query":result["query"]}


def execute_query(state: State):
    """Execute SQL query and return result."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    result = execute_query_tool.invoke(state["query"])
    return {"result":result}


def generate_answer(state: State):
    """Generate answer from result."""
    return {"answer":state["result"]}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"Question: {state['question']}\n"
        f"SQL Query: {state['query']}\n"
        f"SQL Result: {state['result']}"
    )
    
    response = llm.invoke(prompt)
    return {"answer":response.content}



graph_builder = StateGraph(state_schema=State).add_sequence([write_query,execute_query,generate_answer])
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile(checkpointer=memory,interrupt_before=["execute_query"])

for step in graph.stream(
    {"question": "How many employees are there?"},config=config, stream_mode="updates"
):
    print(step)
    
    
try:
    user_approval = input("Do you want to go to execute query? (yes/no): ")
except Exception:
    user_approval = "no"

if user_approval.lower() == "yes":
    # If approved, continue the graph execution
    for step in graph.stream(None, config, stream_mode="updates"):
        print(step)
else:
    print("Operation cancelled by user.")