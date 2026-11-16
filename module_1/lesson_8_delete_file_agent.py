from deepagents import create_deep_agent
from models import model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langchain_core.tools import tool
import questionary

SYSTEM_PROMPT = """You are a filesystem assistant. Your domain is strictly
restricted to filesystem operations. Use the delete_file tool to delete a file.
"""

config = RunnableConfig({"configurable": {"thread_id": "my_thread"}})

@tool
def delete_file(path: str) -> str:
    """Delete a file"""
    return f"The file under {path!r} has been deleted."

def is_risky_folder(req) -> bool:
    is_risky = req.tool_call["args"].get("path", "").startswith("/etc")
    return is_risky

agent = create_deep_agent(
    name="Delete File Agent",
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[delete_file],
    interrupt_on={
        "delete_file": {
            "allowed_decisions": ["approve", "reject"],
            "when": is_risky_folder
        }
    },
    checkpointer=MemorySaver()
)

def main():
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Delete the file /etc/permissions.xml"}
        ]
    }, config=config, version="v2")

    while result.interrupts:
        pending = result.interrupts[0].value
        decisions = []
        for req in pending["action_requests"]:
            if req["name"] == "delete_file":
                file_path = req["args"]["path"]
                action = questionary.select(
                    f"Do you really want to delete {file_path!r}?",
                    choices=["Yes", "No"]
                ).ask()
                if action == "Yes":
                    decisions.append({"type": "approve"})
                else:
                    decisions.append({
                        "type": "reject",
                         "message": f"User rejected deleting {file_path!r}."
                    })
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            version="v2"
        )

    print(result.value["messages"][-1].text)

if __name__ == "__main__":
    main()