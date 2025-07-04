from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational"
)

model = ChatHuggingFace(llm=endpoint)

template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="Write a 5 line summary on the following text. \n {text}",
    input_variables=['text']
)

prompt1 = template1.invoke({'topic': 'black hole'})
response = model.invoke(prompt1)

prompt2 = template2.invoke({'text': response.content})
final_response = model.invoke(prompt2)

print(final_response.content)