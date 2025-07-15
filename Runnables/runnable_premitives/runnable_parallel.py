from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=['topic']
)

model1 = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(prompt1, model1, parser),
        'linkedin': RunnableSequence(prompt2, model2, parser)
    }
)

response = parallel_chain.invoke(
    {'topic': 'AI'}
)

print(response['tweet'])
print()
print(response['linkedin'])