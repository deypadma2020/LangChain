from langchain_core.tools import tool

# step 1 - create a function
def addition(a,b):
    '''
    Addition of two real numbers
    '''
    return a + b

# step 2 - add type hints
def addition(a: int, b: int) -> int:
    '''
    Addition of two real numbers
    '''
    return a + b

# step 3 - add tool decorator
@tool
def addition(a: int, b: int) -> int:
    '''
    Addition of two real numbers
    '''
    return a + b

# step 4 - create an object
response = addition.invoke(
    {
        'a': 26,
        'b': 34
    }
)

# step 5 - print response
print(response)
print()

print(addition.name)
print(addition.description)
print(addition.args)
print()

print(addition.args_schema.model_json_schema())