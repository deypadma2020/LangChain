from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

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