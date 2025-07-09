from langchain_anthropic import ChatAnthropic  # use new package
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()  # optional, if you're using a .env file for the API key

# Initialize the LLM
llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")  # note: use model_name instead of model

# Create prompt template
prompt = PromptTemplate(
    input_variables=['topic'],
    template="Suggest a catchy blog title about {topic}."
)

# Get input
topic = input("Enter a topic: ")
formatted_prompt = prompt.format(topic=topic)

# Generate title
blog_title = llm.invoke(formatted_prompt)

print("Generated Blog Title:", blog_title)
