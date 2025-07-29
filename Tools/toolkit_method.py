from langchain_core.tools import tool

@tool
def addition(a: int, b: int) -> int:
    '''
    Addition of two real integer numbers
    '''
    return a + b

@tool
def subtruction(a: int, b: int) -> int:
    '''
    Substruction of two real integer numbers
    '''
    return a - b

@tool
def multiplication(a: int, b: int) -> int:
    '''
    Multiplication of two real integer numbers
    '''
    return a * b

@tool
def division(a: int, b: int) -> int:
    '''
    Division of two real integer numbers
    '''
    return a / b

class SimpleCalculatorToolKit:
    def get_tools(self):
        return [addition, subtruction, multiplication, division]
    
toolkit = SimpleCalculatorToolKit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, '==>', tool.description)