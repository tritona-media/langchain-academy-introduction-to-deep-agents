from deepagents import create_deep_agent
from models import model

# wraps the model in the deepagents harness
agent = create_deep_agent(model=model)
result = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is an LLM?"}
    ]
})

# "messages" contains a list of all messages in the conversation, including 
# user and AI messages
last_message = result["messages"][-1]

# prints the last message text (which is an AI message)
# ".text" is a convenience property that returns the text of the message
print(last_message.text)

print("---")

# ".content" is the raw content of the message, which may include additional metadata
print(last_message.content)