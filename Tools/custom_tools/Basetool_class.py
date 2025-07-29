from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class AdditionInput(BaseModel):
    a: int = Field(required=True, description="The fist number need to provide")
    b: int = Field(required=True, description="The second number need to provide")

class AdditionTool(BaseTool):
    name: str = "Addition Function"
    description: str = "This func takes two intiger numbers and perform addition operation and return a integer value"
    args_schema: Type[BaseModel] = AdditionInput

    def _run(self, a: int, b: int) -> int:
        return a + b

addition_tool = AdditionTool()

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