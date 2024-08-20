
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith tracking
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Creating chatbot
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Question:{question}")
    ]
)


## streamlit framework 

st.title("Langchain Demo With Google Gemini")
input_text = st.text_input("Search the topic you want")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
output_parser = StrOutputParser()


## chain

chain=prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({'question':input_text}))
