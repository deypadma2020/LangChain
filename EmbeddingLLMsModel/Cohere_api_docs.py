from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

response = embedding.embed_documents(documents)

print(response)
print()

print(str(response))
print()

from pprint import pprint
pprint(response)