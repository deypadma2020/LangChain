from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'Negetive'] = Field(default="Neutral", description='Give the sentiment of the feedback')

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive, negetive or neutral \n{feedback} \n{format_instruction}",
    input_variables=['feedback'],
    partial_variables={
        'format_instruction': pydantic_parser.get_format_instructions()
    }
)

classification_chain = prompt1 | model | pydantic_parser

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | str_parser),
    (lambda x:x.sentiment == 'negetive', prompt3 | model | str_parser),
    RunnableLambda(lambda x: "Neutral")
)

chain = classification_chain | branch_chain

response = chain.invoke({'feedback': "Padma consistently provides excellent answers to all her SQL questions."})

print(response)

chain.get_graph().print_ascii()