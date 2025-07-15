# LangChain Declarative Expression (LCEL) Example

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Define the prompt template
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

# Initialize the LLM
llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

# Define the output parser
parser = StrOutputParser()

# Build the LCEL chain using the pipe operator
chain = prompt | llm | parser

# Invoke the chain with input
response = chain.invoke({"topic": "Mother"})

# Print the final output
print(response)
