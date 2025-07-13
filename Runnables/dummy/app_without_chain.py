from llm_class import DemoLLM
from prompt_template import DemoPromptTemplate

# Instantiate LLM
llm = DemoLLM()

# Instantiate PromptTemplate
template = DemoPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

# Format the prompt
formatted_prompt = template.format({"length": "short", "topic": "India"})

# Get prediction
response = llm.predict(formatted_prompt)

print("Response:", response)
