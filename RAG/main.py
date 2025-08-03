import os
from urllib.parse import urlparse, parse_qs
import yt_dlp
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Step 1: Download audio from YouTube
def download_audio_from_youtube(video_url, output_path='audio.mp3'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    return output_path

# Step 2: Transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result['text']

# Step 3: Format LangChain documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# === MAIN EXECUTION FLOW ===

# Video URL
video_url = "https://www.youtube.com/watch?v=BV0YUeam4y8"
video_id = parse_qs(urlparse(video_url).query).get("v", [""])[0]

# Download audio
print("Downloading audio...")
audio_path = download_audio_from_youtube(video_url)

# Transcribe
print("Transcribing audio...")
transcript = transcribe_audio(audio_path)

# Chunking & Embedding
print("Embedding transcript...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Prompt
prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer the question only using the context below.
If not enough context is available, respond with "I don't know".

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

# LLM - Gemini Pro
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))

# Build LangChain pipeline
chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Ask user question
question = input("Ask a question based on the video: ")
result = chain.invoke(question)
print("\nFinal Answer:\n", result)

