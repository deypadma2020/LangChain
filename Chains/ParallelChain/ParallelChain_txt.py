from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n{text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n{text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quizes into a singe document \nnotes -> {notes} and quizes -> {quizes}",
    input_variables=['notes', 'quizes']
)

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quizes": prompt2 | model | parser
    }
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

with open("Chains/ParallelChain/SVM.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text[:5000]

response = chain.invoke({"text": text})

print(response)

chain.get_graph().print_ascii()
