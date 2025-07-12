from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", temperature=0)

response = llm.invoke("What is the capital of India?")

# print(response)
# print()

print(response.content)
print('############################################')
print()

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

response = llm.invoke("What is the capital of India?")

# print(response)
# print()

print(response.content)
