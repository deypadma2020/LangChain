from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")


response = llm.invoke("What is the capital of India?")

# print(response)
# print()

print(response.content)
