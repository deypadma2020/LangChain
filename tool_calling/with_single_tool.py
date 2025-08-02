from langchain_core.tools import tool

# ---------------------- tool create ----------------------
@tool
def addition(a: int, b: int) -> int:
    '''
    Addition of two real integer numbers
    '''
    return a + b




# ---------------------- tool binding ----------------------
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

llm_with_tools = llm.bind_tools([addition])




# ---------------------- tool calling ----------------------
general_response = llm_with_tools.invoke("Hi! How are you doing")
print("general_response: \n", general_response)
print()

tool_call_response = llm_with_tools.invoke("can you make a addition 2 with 3?")
print("Tool Calling Response: \n", tool_call_response)
print()

print("Extract only tool call part: \n", tool_call_response.tool_calls)
print()
print("Extract the first slice of tool call: \n", tool_call_response.tool_calls[0])
print()
print("Extract the first slice of tool call: \n", tool_call_response.tool_calls[0]['args'])
print()


# ---------------------- tool execution ----------------------

actual_response = addition.invoke(tool_call_response.tool_calls[0]['args'])
print("To the point User query response: \n", actual_response)
print()

actual_structured_response = addition.invoke(tool_call_response.tool_calls[0])
print("formatted User query response: \n", actual_structured_response)
print()

