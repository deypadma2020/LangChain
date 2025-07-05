from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import pdfplumber # type: ignore

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

text = ""
with pdfplumber.open("Chains/Algorithmic Puzzles [Levitin & Levitin 2011-10-14].pdf") as pdf:
    for page in pdf.pages:
        text += page.extract_text() or ""

text = text[:5000]

response = chain.invoke({"text": text})

print(response)

chain.get_graph().print_ascii()