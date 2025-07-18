from langchain_community.document_loaders import PyPDFLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load PDF
loader = PyPDFLoader("RAGComponents/doc/dl-curriculum.pdf")
pages = loader.load()

# Concatenate all page contents into one string
full_text = "\n".join([page.page_content for page in pages])

# Initialize model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Prepare prompt
prompt = PromptTemplate(
    template="Based on the document below, answer the question:\n"
             "What is the biological inspiration behind ANN (Artificial Neural Network)?\n\n"
             "Document:\n{doc}",
    input_variables=['doc']
)

# Set up parser and chain
parser = StrOutputParser()
chain = prompt | model | parser

# Invoke chain with document content
response = chain.invoke({'doc': full_text})

# Print the response
print(response)




# from langchain_community.document_loaders import PyPDFLoader

# loader =  PyPDFLoader("RAGComponents\doc\dl-curriculum.pdf")

# docs = loader.load()

# print(docs)
# print()

# print(len(docs))
# print()

# print(docs[0].page_content)
# print()

# print(docs[0].metadata)

"""
simple clean pdf - PyPDFLoader
PDFs with tables/columns - PDFPlumnberLoader
Scanned/Image PDFs - UnstructuredPDFLoader/AmazonTextractLoader
Need layout and image data - PyMuPDFLoader
Want best structure extraction - UnstructuredPDFLoader
"""