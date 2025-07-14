from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=['text']
)

model1 = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
model2 = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

parser =StrOutputParser()

chain = RunnableSequence(prompt1, model1, parser, prompt2, model2, parser)

response = chain.invoke({'topic': 'AI'})

print(response)