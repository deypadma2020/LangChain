from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

import time
import random

load_dotenv()

# Initialize the model and parser
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
parser = StrOutputParser()

# Prompt Template
prompt = PromptTemplate(
    template="Answer the following question: {question} based on the text:\n{text}",
    input_variables=["question", "text"]
)

# Set up Selenium Chrome driver
options = Options()
options.add_argument('--headless')  # run in background
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# URL to load
url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'

try:
    driver.get(url)
    time.sleep(5)  # allow page to fully render
    page_content = driver.page_source
finally:
    driver.quit()

# Wrap in LangChain Document format
doc = Document(page_content=page_content, metadata={"source": url})

# Build the LCEL chain
chain = prompt | model | parser

# Retry logic for overloaded model
max_retries = 5
for attempt in range(max_retries):
    try:
        response = chain.invoke({
            "question": "What is the product that we are talking about?",
            "text": doc.page_content
        })
        print(response)
        break
    except Exception as e:
        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
        if "overloaded" in str(e).lower():
            wait_time = random.randint(3, 7)
            print(f"Retrying in {wait_time} seconds...\n")
            time.sleep(wait_time)
        else:
            break
