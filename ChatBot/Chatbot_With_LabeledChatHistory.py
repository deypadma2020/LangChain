from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

chat_history = [
    SystemMessage(content="You are a helpful travel guide.")
]

while True:
    user_input =input("You: ")
    chat_history.append(HumanMessage(content=user_input))

    if user_input == 'exit':
        break

    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))

    print("AI: ", response.content)

print(chat_history)