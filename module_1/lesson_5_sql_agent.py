from deepagents import create_deep_agent
from models import model
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from pathlib import Path

DB_PATH = Path(__file__).parent / "chinook.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

SYSTEM_PROMPT = """You are a SQL analyst with access to the Chinook music store database.

Rules:
- Use read_sql for SELECT queries.
- Do not modify the database.
- If a tool returns an error, revise the SQL and try again.
- Show your SQL in your final answer.
"""

@tool
def read_sql(query: str) -> str:
    """Run a read-only SELECT query against the Chinook music store database."""
    try:
        return str(db.run(query))
    except Exception as e:
        return f"Error: {e}"


agent = create_deep_agent(
    name="SQL_Agent",
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[read_sql]
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Which five genres have the most tracks?"
        }
    ]
})

print(result["messages"][-1].text)