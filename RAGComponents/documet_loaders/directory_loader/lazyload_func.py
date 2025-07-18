from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize the model and output parser
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")
parser = StrOutputParser()

# Load PDFs using lazy_load
loader = DirectoryLoader(
    path=r"RAGComponents\books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

# Collect first few pages of text for analysis (avoid long input issues)
extracted_pages = []
max_pages = 10

for i, document in enumerate(docs):
    if i >= max_pages:
        break
    extracted_pages.append(document.page_content)

combined_text = "\n\n".join(extracted_pages)

# Create the prompt template
prompt = PromptTemplate(
    template=(
        "Here is a sample of content extracted from multiple books:\n\n"
        "{book_text}\n\n"
        "Based on this content, what insights can you extract about algorithmic problem-solving techniques?"
    ),
    input_variables=["book_text"]
)

# Create LCEL chain
chain = prompt | model | parser

# Run the chain with combined text
response = chain.invoke({"book_text": combined_text})

# Display the result
print(response)







# from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# loader = DirectoryLoader(
#     path=r"RAGComponents\books",
#     glob="*.pdf",
#     loader_cls=PyPDFLoader 
# )

# # docs = loader.load()

# # for document in docs:
# #     print(document.metadata, end='\n\n')

# docs = loader.lazy_load()

# for document in docs:
#     print(document.metadata, end='\n\n')