from langchain_community.document_loaders import CSVLoader
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Load CSV file
loader = CSVLoader(file_path=r"RAGComponents\doc\Social_Network_Ads.csv")
docs = loader.load()

# Combine all rows into one string
all_rows = "\n\n".join([doc.page_content for doc in docs])

# Set up model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Set up prompt
prompt = PromptTemplate(
    template=(
        "Here is a CSV dataset about users and their ad purchases:\n\n"
        "{data}\n\n"
        "Now answer the following question:\n"
        "How many males under the age of 40 made a purchase?"
    ),
    input_variables=['data']
)

# Set up parser and chain
parser = StrOutputParser()
chain = prompt | model | parser

# Invoke the chain with the CSV data
response = chain.invoke({'data': all_rows})

# Print the output
print(response)







# from langchain_community.document_loaders import CSVLoader

# loader = CSVLoader(file_path=r"RAGComponents\doc\Social_Network_Ads.csv")

# docs = loader.load()

# print(len(docs))
# print()

# print(docs[0])