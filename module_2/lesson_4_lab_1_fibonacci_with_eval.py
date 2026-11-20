from deepagents import create_deep_agent
from models import model
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    name="Fibonacci With Eval Agent",
    model=model,
    middleware=[CodeInterpreterMiddleware()]
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": (
                "Use the eval tool to compute and return the first 15 "
                "Fibonacci numbers."
            )
        }
    ]
})

print(result["messages"][-1].text)
