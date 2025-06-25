import getpass
import os

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")


from langchain_core.messages import HumanMessage, SystemMessage

response_1 = model.invoke("Hello")
response_2 = model.invoke([{"role": "user", "content": "Hello"}])
response_3 = model.invoke([HumanMessage("Hello")])

print(response_1.content)
print(response_2.content)
print(response_3.content)
