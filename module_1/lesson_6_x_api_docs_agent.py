from deepagents import create_deep_agent
from models import model
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

ALLOWED_TOOLS = {"search_x"}

async def main():
    client = MultiServerMCPClient({
        "x-api-docs": {
            "transport": "http",
            "url": "https://docs.x.com/mcp"
        }
    })

    tools = [t for t in await client.get_tools() if t.name in ALLOWED_TOOLS]

    agent = create_deep_agent(
        name="X API Docs Agent",
        model=model,
        system_prompt="Use the X MCP Tool to search across the X knowledge "
        "base to find relevant information, code examples, API references, and "
        "guides.",
        tools=tools
    )

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "Wie filtere ich bei der X-Suche Retweets heraus "
                 "und frage gleichzeitig die Profilbilder der Autoren ab?"
            }
        ]
    })

    print(result["messages"][-1].text)
if __name__ == "__main__":
    asyncio.run(main())