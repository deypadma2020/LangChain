import getpass
import os
import asyncio

if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")
##################################################################


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

##################################################################

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


#####################################################################

from langchain_core.messages import SystemMessage, trim_messages, HumanMessage, AIMessage

trimmer = trim_messages(
    max_tokens = 512,
    strategy = "last",
    token_counter = model,
    # include_system = True,
    allow_partial = False,
    start_on = "human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

language = "English"


#####################################################################
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

async def main():


    # define the function taht calls the model
    async def call_model(state: State):
        trimmed_messages = await trimmer.ainvoke(state["messages"])
        prompt = await prompt_template.ainvoke(
            {
                "messages": trimmed_messages,
                "language": state["language"]
            }
        )
        response = await model.ainvoke(prompt)
        return {"messages": [response]}


    # define a new graph
    workflow = StateGraph(state_schema=State)

    # define the(single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

######################################################################

    # creating config
    config = {
        "configurable": 
        {
            "thread_id": "Thread-1"
        }
    }

    query = "What is my name?"
    input_messages = messages + [HumanMessage(query)]
    output = await app.ainvoke(
        {
            "messages": input_messages,
            "language": language
        }, 
        config
    )
    output["messages"][-1].pretty_print()

######################################################################

    # creating config
    config = {
        "configurable": 
        {
            "thread_id": "Thread-2"
        }
    }

    query = "What math problem did I ask?"
    input_messages = messages + [HumanMessage(query)]
    output = await app.ainvoke(
        {
            "messages": input_messages,
            "language": language
        }, 
        config
    )
    output["messages"][-1].pretty_print()




    # creating config
    config = {
        "configurable": 
        {
            "thread_id": "Thread-3"
        }
    }

    query = "Do you know, who I am?"
    input_messages = messages + [HumanMessage(query)]
    output = await app.ainvoke(
        {
            "messages": input_messages,
            "language": language
        }, 
        config
    )
    output["messages"][-1].pretty_print()

if __name__=="__main__":
    asyncio.run(main())

