from deepagents import create_deep_agent
from models import model
from langchain_core.tools import tool
import resend
import os

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient with the given subject and body.
    
    Args:
        to: The recipient's email address.
        subject: The subject of the email.
        body: The body content of the email.

    Returns:
        A string indicating the result of the email sending operation.
    """
    resend.api_key = os.getenv("RESEND_API_KEY")
    try:
        resend.Emails.send({
            "from": "onboard@resend.dev",
            "to": to,
            "subject": subject,
            "html": body
        })
        return f"Email sent to {to} with subject '{subject}' and body '{body}'."
    except Exception as e:
        return f"Error sending email to {to}: {e}"

agent = create_deep_agent(
    name="Email_Agent",
    model=model,
    system_prompt="You are an email assistant. You can send emails using the send_email tool.",
    tools=[send_email]
)

def main():
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Send an email to tritona.media@gmail.com with some facts about FC Bayern and a subject line of 'FC Bayern Facts'."
            }
        ]
    })
    print(result["messages"][-1].text)

if __name__ == "__main__":
    main()