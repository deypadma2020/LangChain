from llm_class import DemoLLM
from prompt_template import DemoPromptTemplate
from runnable_connector import RunnableConnector
from str_output_parser import DemoStrOutputParser


# Instantiate PromptTemplate
template1 = DemoPromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

template2 = DemoPromptTemplate(
    template="Explain the following joke {response}",
    input_variables=["response"]
)

# Instantiate LLM
llm = DemoLLM()

# Intantiate String Output Parser
parser = DemoStrOutputParser()

chain1 = RunnableConnector([template1, llm])
# respomse1 = chain1.invoke({"length": "short", "topic": "India"})
# print('respomse: ', respomse1)

chain2 = RunnableConnector([template2, llm, parser])
# respomse2 = chain2.invoke({"response": "This is a joke explanation"})
# print('respomse: ', respomse2)

final_chain = RunnableConnector([chain1, chain2])
response = final_chain.invoke({'topic': 'cricket'})
print("Joke explanation: ", response)