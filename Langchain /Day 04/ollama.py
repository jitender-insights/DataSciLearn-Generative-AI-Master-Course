
# from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama


import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith tracking
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")



# Creating chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Question:{question}")
    ]
)


## streamlit framework 

st.title("Langchain Demo With Ollama")
input_text = st.text_input("Search the topic you want")
llm = Ollama(model="gemma2:2b")
output_parser = StrOutputParser()


## chain

chain=prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({'question':input_text}))
