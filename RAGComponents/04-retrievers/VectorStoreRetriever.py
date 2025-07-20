from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

documents_ = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=documents_,
    embedding=embedding_model,
    collection_name="my_collection",  
    persist_directory=r"RAGComponents\04-retrievers\vectorstore"  
)


retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is chroma db used for?"
response_docs = retriever.invoke(query)

for i, doc in enumerate(response_docs):
    print(f"\n--- Response {i+1} ---")
    print(f"Content:\n{doc.page_content}...")