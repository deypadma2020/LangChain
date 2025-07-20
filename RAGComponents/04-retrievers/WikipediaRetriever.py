from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang="en")

query = "the geopolitical history of india and pakistan from the perspective of a chinese"

response_docs = retriever.invoke(query)

for i, doc in enumerate(response_docs):
    print(f"\n--- Response {i+1} ---")
    print(f"Content:\n{doc.metadata}...\n")
    print(f"Content:\n{doc.page_content}...")