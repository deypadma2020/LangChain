from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# Save to disk if needed
vectorstore.save_local("RAGComponents/04-retrievers/vectorstore/faiss_store")

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5 # "lambda_mult": 1 -> behave like as same as regular retriver
    }
)

query = "What is langchain?"
response_docs = retriever.invoke(query)

for i, doc in enumerate(response_docs):
    print(f"\n--- Response {i+1} ---")
    print(f"Content:\n{doc.page_content}...")