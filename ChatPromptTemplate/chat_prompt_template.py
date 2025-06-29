from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.messages import SystemMessage, HumanMessage

# chat_template = ChatPromptTemplate([
#     SystemMessage(content="You are a helpful {domain} expert!"),
#     HumanMessage(content="Explain in simple terms, what is {topic}")
# ])
# -------------------------This way not work -------------------------

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert!"),
    ("human", "Explain in simple terms, what is {topic}")
])

prompt = chat_template.invoke({'domain':'Painter', 'topic': 'Oil Painting'})

print(prompt)

# -------------------------Will get same response-------------------------

chat_template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful {domain} expert!"),
    ("human", "Explain in simple terms, what is {topic}")
])

prompt = chat_template.invoke({'domain':'Painter', 'topic': 'Oil Painting'})

print(prompt)

# Designed for multi-turn chat-based models (like ChatGPT or Claude). It allows you to define roles (system, user, assistant) and messages in a structured way, mimicking real conversations.