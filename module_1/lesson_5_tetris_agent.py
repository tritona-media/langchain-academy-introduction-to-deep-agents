from deepagents import create_deep_agent
from models import model
from langchain_core.tools import tool

@tool
def get_tetris_fact(fact: str) -> str:
    """
    Get a fact about Tetris. Known facts include: published, author, 
    amount_of_bricks, record_score, best_player.

    Args:
        fact (str): The fact to get about Tetris.

    Returns:
        str: A fact about Tetris.
    """
    facts = {
        "published": 1984,
        "author": "Alexey Pajitnov.",
        "amount_of_bricks": 7,
        "record_score": 999999,
        "best_player": "Jonas Neubauer",
    }
    
    return str(facts.get(fact, "Unknown Tetris fact."))

agent = create_deep_agent(
    name="Tetris_Agent",
    model=model,
    system_prompt="""You are a Tetris assistant. Your domain is strictly:
        Tetris facts. Strictly use get_tetris_fact to retrieve
        facts about Tetris. Do not use your own knowledge. If you
        don't know the answer, say 'I don't know'.""",
    tools=[get_tetris_fact]
)

def main():
    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "How many bricks are in Tetris?"
            }
        ]
    })
    print(response["messages"][-1].text)

if __name__ == "__main__":
    main()