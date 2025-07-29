from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class AdditionInput(BaseModel):
    a: int = Field(required=True, description="The fist number need to provide")
    b: int = Field(required=True, description="The second number need to provide")

def addition(a: int, b: int) -> int:
    '''
    Addition of two real numbers
    '''
    return a + b

addition_tool = StructuredTool.from_function(
    func=addition,
    name="Addition Function",
    description="This func takes two intiger numbers and perform addition operation and return a integer value",
    args_schema=AdditionInput
)

response = addition_tool.invoke(
    {
        'a': 26,
        'b': -60
    }
)

print(response)
print()

print(addition_tool.name)
print(addition_tool.description)
print()

print(addition_tool.args)