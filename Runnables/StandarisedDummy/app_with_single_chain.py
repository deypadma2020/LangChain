from llm_class import DemoLLM
from prompt_template import DemoPromptTemplate
from runnable_connector import RunnableConnector
from str_output_parser import DemoStrOutputParser


# Instantiate PromptTemplate
template = DemoPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["length", "topic"]
)

# Instantiate LLM
llm = DemoLLM()

# Intantiate String Output Parser
parser = DemoStrOutputParser()

chain = RunnableConnector([template, llm, parser])
respomse = chain.invoke({"length": "short", "topic": "India"})
print('respomse: ', respomse)