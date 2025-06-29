from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

messages = [
    SystemMessage(content="You are a helpful travel guide."),
    HumanMessage(content="Suggest 5 places in India where I can experience heavy snowfall.")
]

response = model.invoke(messages)

messages.append(AIMessage(content=response.content))

print(messages)