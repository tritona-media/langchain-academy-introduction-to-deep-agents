from langchain_mcp_adapters.client import MultiServerMCPClient
from models import model
from deepagents import create_deep_agent
import asyncio

ALLOWED_TOOLS = {"ask_wiki_question"}

async def main():
    client = MultiServerMCPClient({
        "deep_wiki": {
            "transport": "http",
            "url": "https://mcp.deepwiki.com/mcp"
        }
    })

    tools = [t for t in await client.get_tools() if t.name in ALLOWED_TOOLS]

    agent = create_deep_agent(
        name="DeepWiki GitHub Agent",
        model=model,
        system_prompt="""Use the DeepWiki MCP Tool to answer any question 
                        about a GitHub repository's codebase.""",
        tools=tools
    )

    result = await agent.ainvoke({
        "messages": [
            {
                "role": "user",
                "content": "Wie funktioniert das Reconciler-System im GitHub Projekt facebook/react?"
            }
        ]
    })

    print(result["messages"][-1].text)

if __name__ == "__main__":
    asyncio.run(main())