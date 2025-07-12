from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Load the document
loader = TextLoader("Runnables/SVM.txt")
documents = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Use HuggingFace embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in Chroma vectorstore
vectorstore = Chroma.from_documents(docs, embedding_model)

# Create retriever
retriever = vectorstore.as_retriever()

# Define query
query = "What are the key takeaways from the document?"
retrieved_docs = retriever.invoke(query)  # updated method

# Combine documents
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Use Claude 3.5 Sonnet via correct ChatAnthropic
llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0.7)

# Generate prompt
prompt = f"Based on the following text, answer the question: {query}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

# Print answer
print("Answer:", answer)




"""---

**PDF Query Application with ChromaDB and LLM**

This application allows users to query the contents of a PDF file using semantic search and an LLM (Large Language Model). The workflow includes:

1. **PDF Ingestion & Processing**
   The application reads and splits the PDF into manageable text chunks using a recursive text splitter.

2. **Embedding & Storage**
   Each chunk is converted into vector embeddings using Hugging Face models and stored in **ChromaDB**, a persistent vector store.

3. **Semantic Search**
   When a user submits a question, the application retrieves semantically relevant chunks from the ChromaDB vector store.

4. **LLM Response Generation**
   The retrieved text chunks are passed to an LLM (e.g., Claude or GROQ's models) along with the user's question. The LLM generates a concise, context-aware answer.

This pipeline enables intelligent, natural language querying of PDF documents with precise and relevant answers.

---

"""