import getpass
import os

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")



from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# define a new graph
workflow = StateGraph(state_schema=MessagesState)

# define the function taht calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

# define the(single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# creating config
config = {"configurable": {"thread_id": "Thread-1"}}


from langchain_core.messages import HumanMessage, AIMessage

query = "Hi! dear, I am Padma."
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()



query = "Do you know, what's my name?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()




# creating config
config = {"configurable": {"thread_id": "Thread-2"}}

query = "Do you know, who I am?"
input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()
