from deepagents import create_deep_agent
from models import model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

config = RunnableConfig({"configurable": {"thread_id": "my_thread"}})

agent = create_deep_agent(
    name="Time Travel Agent",
    model=model,
    checkpointer=MemorySaver()
)

def main():
    agent.invoke({
        "messages": [{"role": "user", "content": "My name is John."}]
    }, config=config)

    agent.invoke({
        "messages": [{"role": "user", "content": "I am 30 years old."}]
    }, config=config)

    agent.invoke({
        "messages": [{"role": "user", "content": "I live in London."}]
    }, config=config)

    snapshots = list(agent.get_state_history(config))

    turns = [s for s in snapshots if s.next == ()]

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Who am I and how old am I?"}]
    }, config=turns[1].config)

    print(result["messages"][-1].text)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Where do I live?"}]
    }, config=turns[1].config)

    print(result["messages"][-1].text)

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Where do I live?"}]
    }, config=turns[0].config)

    print(result["messages"][-1].text)

if __name__ == "__main__":
    main()