from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

response = embedding.embed_documents(documents)

print(response)
