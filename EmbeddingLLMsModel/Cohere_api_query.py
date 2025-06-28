from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY")
)

response = embedding.embed_query("Delhi is the capital of India")

print(response)



# ------------------------------------------------
# # Using Monkey Patch

# from langchain_cohere import CohereEmbeddings
# from dotenv import load_dotenv

# import os
# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") # monkey-patch env

# load_dotenv()

# embedding = CohereEmbeddings(model="embed-english-v3.0")

# response = embedding.embed_query("Delhi is the capital of India")

# print(response)