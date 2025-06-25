import getpass
import os

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")



from langchain_core.messages import HumanMessage, AIMessage

response = model.invoke([
    HumanMessage(content="Hi! dear, I am Padma."),
    AIMessage(content="Hello Padma! Nice to meet you. I'm Claude, an AI assistant. How can I help you today?"),
    HumanMessage(content="Do you know, what's my name?")
])

print(response.content)