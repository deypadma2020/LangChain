from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

st.header("Research Paper Summarization Tool")

user_input = st.text_input("Enter your query:")

if st.button("Summarize"):
    response = model.invoke(user_input)
    st.write(response.content)


# python -m streamlit run Build_a_Simple_LLM_APP_with_UI\Anthropic_api_app.py
