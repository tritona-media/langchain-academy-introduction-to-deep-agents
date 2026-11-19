from deepagents import create_deep_agent
from deepagents.backends.langsmith import LangSmithSandbox
from models import model
from langsmith.sandbox import SandboxClient
from pathlib import Path

client = SandboxClient()
ls_sanbox = client.create_sandbox(name=f"my-sandbox")
backend = LangSmithSandbox(sandbox=ls_sanbox)

agent = create_deep_agent(
    name="Homework Agent",
    model=model,
    backend=backend,
    system_prompt=(
        "You are a helpful assistant that has access to an external LangSmith sandbox. "
        "If necessary: write Python code to a file within the sandbox before running it. "
        "Install packages with 'pip install --break-system-packages <package>' "
        "if you have to. Example: 'pip install --break-system-packages matplotlib'. "
        "Use matplotlib when you need to create a chart."
    )
)

try:
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Create a file /football_clubs_members_amount.csv that "
                    "contains a list of the top ten "
                    "European football clubs that have the most club members "
                    "in descending order. Use this format: <club>,<amount> "
                    "and put each club in a separate line."
                )
            }
        ]
    })
    print(result["messages"][-1].text)

    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read the file /football_clubs_members_amount.csv and "
                    "create a bar chart depicting the football clubs and the "
                    "amount of their members. Save the chart to "
                    "/football_clubs_members_amount.png."
                )
            }
        ]
    })
    print(result["messages"][-1].text)

    chart_bytes = ls_sanbox.read("/football_clubs_members_amount.png")
    out_path = Path(__file__).parent / "football_clubs_members_amount.png"
    out_path.write_bytes(chart_bytes)
finally:
    client.delete_sandbox(ls_sanbox.name)