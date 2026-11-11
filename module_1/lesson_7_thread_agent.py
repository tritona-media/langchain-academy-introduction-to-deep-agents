from deepagents import create_deep_agent
from models import model
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
import pprint

config: RunnableConfig = {"configurable": {"thread_id": "my_thread"}}
config_2: RunnableConfig = {"configurable": {"thread_id": "another_thread"}}

agent = create_deep_agent(
    name="Thread Agent",
    model=model,
    checkpointer=MemorySaver()
)

agent.invoke({
    "messages": [
        {"role": "user", "content": "My favorite color is blue."}
    ]
}, config=config)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is my favorite color?"}
    ]
}, config=config)

print(result["messages"][-1].text)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is my favorite color?"}
    ]
}, config=config_2)

print(result["messages"][-1].text)