from deepagents import create_deep_agent
from models import model
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

config_a = RunnableConfig({"configurable": {"thread_id": "thread_a"}})
config_b = RunnableConfig({"configurable": {"thread_id": "thread_b"}})

agent_a = create_deep_agent(
    name="Agent A",
    model=model,
    checkpointer=MemorySaver()
)

agent_a.invoke({
    "messages": [{"role": "user", "content": "My favorite car brand is BMW."}]
}, config=config_a)

result = agent_a.invoke({
    "messages": [{"role": "user", "content": "What is my favorite car brand?"}]
}, config=config_a)

# Knows my favorite car brand
print(result["messages"][-1].text)

result = agent_a.invoke({
    "messages": [{"role": "user", "content": "What is my favorite car brand?"}]
}, config=config_b)

# Does not know my favorite car brand
print(result["messages"][-1].text)

agent_b = create_deep_agent(
    name="Agent B",
    model=model,
    checkpointer=MemorySaver()
)

result = agent_b.invoke({
    "messages": [{"role": "user", "content": "What is my favorite car brand?"}]
}, config=config_a)

# Does not know my favorite car brand
print(result["messages"][-1].text)
