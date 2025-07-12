from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load the LLM()
# llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620", temperature=0.7)
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

# Create a PromptTemplate
prompt = PromptTemplate(
    input_variables=["topic"], # Define what input is needed
    template="Suggest a catchy blog title about {topic}."
)

# Create an LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain with a specific topic
topic = input("Enter a topic: ")
output = chain.run(topic)

print("Generated Blog Title: ", output)








# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate

# from langchain_core.runnables import RunnableSequence

# # Load .env variables
# load_dotenv()

# # Initialize LLM
# llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)

# # Define Prompt
# prompt = PromptTemplate.from_template("Suggest a catchy blog title about {topic}.")

# # Build a RunnableChain
# chain = prompt | llm

# # Get input and invoke
# topic = input("Enter a topic: ")
# output = chain.invoke({"topic": topic})

# print("Generated Blog Title:", output.content)



"""
This script generates catchy blog titles using a user-provided topic. 
It uses a language model (Claude or GROQ’s DeepSeek) via LangChain, 
with a prompt template and LLMChain to structure the input and 
produce creative title suggestions. 
An alternative version using `RunnableSequence` is also included for modern LangChain usage.
"""
