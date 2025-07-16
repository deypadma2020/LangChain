from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following text -\n {text}",
    input_variables=['text']
)

model1 = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)
model2 = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

parser = StrOutputParser()

report_generation_chain = prompt1 | model1 | parser

branch_chain = RunnableBranch(
    (
        lambda x: len(x.split()) > 300, 
        prompt2 | model2 | parser
    ),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)

response = final_chain.invoke({'topic': 'Russia vs Ukraine'})

print(response)