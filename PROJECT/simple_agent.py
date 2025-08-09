from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import requests

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

# --- Tool Definitions ---

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    Fetch the currency conversion factor between a base currency and a target currency.
    """
    url = f"https://v6.exchangerate-api.com/v6/06ff5a588198d533c805437e/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    data = response.json()
    return data.get("conversion_rate")

@tool
def convert(base_currency_value: float, conversion_rate: float) -> float:
    """
    Convert a base currency value to the target currency using the conversion rate.
    """
    return base_currency_value * conversion_rate

# --- LLM Setup ---

llm = ChatGroq(model="llama3-70b-8192", temperature=0)

# --- Initialize the Agent ---

agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Run the Agent ---

user_query = "Please fetch the conversion rate between INR and USD, and then convert 10 INR to USD using that rate."
response = agent_executor.invoke({"input": user_query})

print("\n== Final Answer ==")
print(response["output"])
