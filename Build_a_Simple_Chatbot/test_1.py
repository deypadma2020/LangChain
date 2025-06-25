import getpass
import os

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")



from langchain_core.messages import HumanMessage

response_1 = model.invoke([HumanMessage(content="Hi! dear, I am Padma.")])
print(response_1.content)
print()

response_2 = model.invoke([HumanMessage(content="Do you know, what's my name?")])
print(response_2.content)