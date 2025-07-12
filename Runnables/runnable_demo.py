from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain.schema.runnable import Runnable

load_dotenv()

model = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

print(model.get_input_jsonschema())
print(model.get_output_jsonschema())