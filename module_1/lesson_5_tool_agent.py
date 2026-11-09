from deepagents import create_deep_agent
from models import model
from langchain_core.tools import tool
import random

@tool
def roll_dice(sides: int = 6) -> int:
    """Roll a dice with the given number of sides and return the result."""
    return random.randint(1, sides)

agent = create_deep_agent(
    model=model,
    tools=[roll_dice]
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Roll a dice with 20 sides."
    }]
})

print(result["messages"][-1].text)
print("---")
print(result["messages"])