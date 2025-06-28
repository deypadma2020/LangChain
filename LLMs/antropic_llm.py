from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")  

response = llm.invoke("What is the capital of India?")

print(response)
print()

print(response.content)
