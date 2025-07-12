# from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
# from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS


load_dotenv()

# Load the document
loader = TextLoader(r"Runnables\SVM.txt") # Ensure SVM.txt exists
documents = loader.load()

# Split the text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Convert text into embeddings & store in Chroma(Vector DB)
vectorstore = FAISS.from_documents(docs, CohereEmbeddings(model="embed-english-v3.0"))

# Create a retriever (this fetches relevant documents)
retriever = vectorstore.as_retriever()

# Initialize LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

# Create RetrievalQAChain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question
query = "What are the key takeaways from the document?"
answer = qa_chain.run(query)

print("Answer: ", answer)



"""
This script implements a RetrievalQA pipeline that allows users to ask questions about a text file (`SVM.txt`). It performs the following steps:

1. Loads and splits the document into chunks.
2. Converts chunks into embeddings using Cohere.
3. Stores embeddings in a FAISS vector store.
4. Retrieves relevant chunks via semantic search.
5. Uses Groq’s DeepSeek model to generate a concise answer.

The system enables natural language querying over the document using LLM-powered answers.
"""