from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=r"RAGComponents\books",
    glob="*.pdf",
    loader_cls=PyPDFLoader 
)

docs = loader.load()

print(len(docs))
print()

print(docs[10].metadata)
print()

print(docs[10].page_content)