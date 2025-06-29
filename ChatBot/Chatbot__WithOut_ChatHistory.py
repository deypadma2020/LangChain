from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

while True:
    user_input =input("You: ")
    if user_input == 'exit':
        break
    response = model.invoke(user_input)
    print("AI: ", response.content)