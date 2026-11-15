from deepagents import create_deep_agent
from models import model
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

SYSTEM_PROMPT = """Your are a helpful assistant that can send emails.

Rules:
- Use send_email when the user asks you to send an email.
- Keep emails concise and professional.
- Do not claim an email was sent until the tool result confirms it.
- When confirming an email was sent, quote the subject and body from the tool
  result, not the original request.
- Don't ask further questions after you've done your job.
"""

config = RunnableConfig({"configurable": {"thread_id": "my_thread_v5"}})

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {to} with subject {subject!r}. Body: {body}"

agent = create_deep_agent(
    name="HITL Agent - v5",
    model=model,
    tools=[send_email],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=MemorySaver(),
    interrupt_on={"send_email": True}
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Send an email to jane@example.com saying I will be 10 minutes late."}]
}, config=config, version="v2")

while result.interrupts:
    print("----------")
    print(result.interrupts)
    print("----------")
    pending = result.interrupts[0].value
    decisions = []
    for req in pending["action_requests"]:
        print(f"\nApproval required for {req['name']}:")
        print(req["args"])

        choice = input("\nApprove, edit, or reject? (approve/edit/reject): ").strip().lower()
        if choice in ("approve", "yes", "y"):
            decisions.append({"type": "approve"})
        elif choice in ("edit", "e"):
            edited_args = dict(req["args"])
            edited_args["body"] = input("New email body: ")
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {"name": req["name"], "args": edited_args}
                }
            )
        else:
            decisions.append({
                "type": "reject",
                "message": "User rejected this email draft."
            })

    result = agent.invoke(Command(resume={"decisions": decisions}), config=config, version="v2")

print(result.value["messages"][-1].text)