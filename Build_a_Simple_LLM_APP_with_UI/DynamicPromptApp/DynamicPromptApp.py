from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt(r'Build_a_Simple_LLM_APP_with_UI\DynamicPromptApp\PromptTemplate\template.json')

prompt = template.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })

if st.button("Summarize"):
    response = model.invoke(prompt)
    st.write(response.content)
