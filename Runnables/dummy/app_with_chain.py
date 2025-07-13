from llm_class import DemoLLM
from prompt_template import DemoPromptTemplate
from chain import DemoLLMChain

# Instantiate LLM
llm = DemoLLM()

# Instantiate PromptTemplate
template = DemoPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

chain = DemoLLMChain(llm, template)
respomse = chain.run({"length": "short", "topic": "India"})
print('respomse: ', respomse)