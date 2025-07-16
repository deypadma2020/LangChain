from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke about {joke}",
    input_variables=['joke']
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

parser = StrOutputParser()

def word_count(text):
    return len(text.split())

joke_generation_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'word_count': RunnableLambda(word_count)
    }
)

final_chain = RunnableSequence(joke_generation_chain, parallel_chain)

response = final_chain.invoke({'joke': 'AI'})

print(
    "{} \n word count - {}".format(response['joke'], response['word_count'])
)