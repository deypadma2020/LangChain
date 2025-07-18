from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=r"RAGComponents\books",
    glob="*.pdf",
    loader_cls=PyPDFLoader 
)

# docs = loader.load()

# for document in docs:
#     print(document.metadata, end='\n\n')

docs = loader.lazy_load()

for document in docs:
    print(document.metadata, end='\n\n')