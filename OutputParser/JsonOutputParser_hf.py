from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational"
)

model = ChatHuggingFace(llm=endpoint)
parser = JsonOutputParser()
template = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic'],
    partial_variables={
        'format_instruction' : parser.get_format_instructions()
    }
)

chain = template | model | parser

response = chain.invoke({"topic": 'black hole'})

print(response)