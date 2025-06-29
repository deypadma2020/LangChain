from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

response = model.invoke("Suggest 5 places to visit in India during December with a budget of ₹15,000.")

print(response.content)
print('---------------------------------------')

model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.5)
response = model.invoke("Suggest 5 places to visit in India during December with a budget of ₹15,000.")
print(response.content)
print('---------------------------------------')

model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=1.0)
response = model.invoke("Suggest 5 places to visit in India during December with a budget of ₹15,000.")
print(response.content)

