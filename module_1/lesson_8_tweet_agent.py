from deepagents import create_deep_agent
from models import model
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
import questionary

config = RunnableConfig({"configurable": {"thread_id": "thread_1"}})

SYSTEM_PRMOPT="""You are a tweet assistant. Your goal is to write a short
and concise tweet about a topic. Use the post_tweet tool to do that.
"""

@tool
def post_tweet(title: str, text: str) -> str:
    """Post a tweet with a title and a text."""
    return f"Tweet was posted with title {title!r} and {text!r}."

agent = create_deep_agent(
    name="Tweet Agent",
    model=model,
    system_prompt=SYSTEM_PRMOPT,
    tools=[post_tweet],
    interrupt_on={"post_tweet": {"allowed_decisions": ["approve", "edit", "reject"]}},
    checkpointer=MemorySaver()
)

def main():
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "Write a tweet about FC Bayern winning Champions League 2027."}
        ]
    }, config=config, version="v2")

    while result.interrupts:
        pending = result.interrupts[0].value
        decisions = []
        for req in pending["action_requests"]:
            if req["name"] == "post_tweet":
                print(f"Approval required for {req["name"]}:")
                print(req["args"])
                action = questionary.select(
                    "How do you want to proceed?",
                    choices=["Approve", "Edit", "Reject"]
                ).ask()
                if action == "Approve":
                    decisions.append({"type": "approve"})
                elif action == "Edit":
                    edited_args = dict(req["args"])
                    edited_args["text"] = questionary.text("New text: ").ask()
                    decisions.append({
                        "type": "edit",
                        "edited_action": {"name": req["name"], "args": edited_args}
                    })
                else:
                    decisions.append({
                        "type": "reject",
                        "message": "Posting tweet has been rejected by user."
                    })
        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            version="v2"
        )

    print(result.value["messages"][-1].text)

if __name__ == "__main__":
    main()