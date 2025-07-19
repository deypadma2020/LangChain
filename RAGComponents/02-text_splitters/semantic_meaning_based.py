from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Loads COHERE_API_KEY from .env

# Initialize the Cohere embedding model
embedding_model = CohereEmbeddings(model="embed-english-v3.0")

# Create the SemanticChunker with CohereEmbeddings
text_splitter = SemanticChunker(
    embedding_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

# Sample text input
sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# Split into semantically meaningful chunks
docs = text_splitter.create_documents([sample])

# Output the number and contents of chunks
print(len(docs))
for i, doc in enumerate(docs, 1):
    print(f"\n--- Chunk {i} ---\n{doc.page_content}")
