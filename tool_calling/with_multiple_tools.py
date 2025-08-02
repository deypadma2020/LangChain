from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

# ---------------------- tool create ----------------------
from langchain_core.tools import tool

@tool
def addition(a: int, b: int) -> int:
    """Adds two integers and returns the result."""
    return a + b

@tool
def subtruction(a: int, b: int) -> int:
    """Subtracts b from a and returns the result."""
    return a - b

@tool
def multiplication(a: int, b: int) -> int:
    """Multiplies two integers and returns the result."""
    return a * b

@tool
def division(a: int, b: int) -> float:
    """Divides a by b and returns the result as a float."""
    return a / b


# ---------------------- tool binding ----------------------
load_dotenv()
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools([addition, subtruction, multiplication, division])

# ---------------------- tool calling loop ----------------------
from langchain_core.messages import HumanMessage

user_query = "can you make a addition 2 with 3; then deduct 4 from the previous result?"

# First user message
messages = [HumanMessage(content=user_query)]

while True:
    response = llm_with_tools.invoke(messages)
    print("\nClaude Response:\n", response.content)

    # If no tool calls, we're done
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        break

    # Execute each tool call and collect ToolMessages
    tool_messages = []
    for tool_call in response.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        if name == "addition":
            result = addition.invoke(args)
        elif name == "subtruction":
            result = subtruction.invoke(args)
        elif name == "multiplication":
            result = multiplication.invoke(args)
        elif name == "division":
            result = division.invoke(args)
        else:
            raise ValueError(f"Unknown tool: {name}")

        print(f"Tool '{name}' executed with args {args} → Result: {result}")
        tool_messages.append(ToolMessage(tool_call_id=tool_call["id"], content=str(result)))

    # Update messages with last response and tool messages
    messages.append(response)
    messages.extend(tool_messages)

# Final result shown after all tool calls are handled
print("\nFinal Answer:\n", response.content)
