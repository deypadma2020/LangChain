from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Load PDFs from the books directory
loader = DirectoryLoader(
    path=r"RAGComponents\books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

# Combine content from the first few pages to avoid exceeding token limits
# Adjust the slice if needed for more/less content
combined_text = "\n\n".join([doc.page_content for doc in docs[:15]])

# LLM setup
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
parser = StrOutputParser()

# Prompt setup
prompt = PromptTemplate(
    template=(
        "You are given content extracted from technical books:\n\n"
        "{book_content}\n\n"
        "Answer the following question based on the given text:\n"
        "How does the book describe the process of building a machine learning system?"
    ),
    input_variables=["book_content"]
)

# LCEL chain
chain = prompt | model | parser

# Invoke with content
response = chain.invoke({"book_content": combined_text})

# Output the result
print(response)








# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# loader = DirectoryLoader(
#     path=r"RAGComponents\books",
#     glob="*.pdf",
#     loader_cls=PyPDFLoader 
# )

# docs = loader.load()

# print(len(docs))
# print()

# print(docs[10].metadata)
# print()

# print(docs[10].page_content)