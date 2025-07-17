from langchain_community.document_loaders import TextLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

prompt = PromptTemplate(
    template="Write a summary for the following poem -\n {poem}",
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader("RAGComponents\doc\SVM.txt", encoding='utf-8')
docs = loader.load()

chain = prompt | model | parser
response = chain.invoke(
    {
        'poem': docs[0].page_content
    }
)

print(response)

