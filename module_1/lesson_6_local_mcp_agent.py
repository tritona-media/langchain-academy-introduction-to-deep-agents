from deepagents import create_deep_agent
from models import model
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

import sys, asyncio

SERVER = Path(__file__).parent / "lesson_6_dice_server.py"

async def main():
    client = MultiServerMCPClient({
        "dice": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVER)]
        }
    })
    tools = await client.get_tools()
    print("Tools from the server: ", [t.name for t in tools])

    agent = create_deep_agent(
        name="Local MCP Agent",
        model=model,
        tools=tools
    )

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "Roll a 20-sided die for me."
            }
        ]
    })
    print("Final answer: ", result["messages"][-1].text)

asyncio.run(main())