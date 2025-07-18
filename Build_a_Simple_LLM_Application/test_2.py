import getpass
import os

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!")
]

response = model.invoke(messages)
print(response.content)
