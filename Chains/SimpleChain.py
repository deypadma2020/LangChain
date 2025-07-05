from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = "Generate 5 interesting facts about {topic}",
    input_variables=['topic']
)

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({'topic': "Mother"})

print(response)

chain.get_graph().print_ascii()