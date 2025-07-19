from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("RAGComponents\doc\dl-curriculum.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_documents(docs)
print(len(result))
print()

print(result[0].metadata)
print()

print(result[0].page_content)