from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Chat Template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent!'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = list()

# Load Chat History
with open(r'MessagePlaceHolder\ChatHistory.txt') as f:
    chat_history.extend(f.readlines())

print("Chat History: \n", chat_history)
print()

# Creat Prompt
prompt = chat_template.invoke({
    'chat_history': chat_history, 
    'query': 'Where is my refund?'
})

print(prompt)