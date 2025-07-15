from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {joke}",
    input_variables=['joke']
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=['text']
)

model1 = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

parser = StrOutputParser()

joke_generation_chain = RunnableSequence(prompt1, model1, parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explanation': RunnableSequence(prompt2, model2, parser)
    }
)

final_chain = RunnableSequence(joke_generation_chain, parallel_chain)

response = final_chain.invoke({'joke': 'Cyber Security'})
print(response)
