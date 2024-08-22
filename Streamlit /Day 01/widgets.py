import streamlit as st 
import pandas as pd

st.title("Streamlit Text Input")
name = st.text_input("Search the topic you want")
if name:
    st.write(f"Hello",{name})
    
emi = st.slider("Select your EMI Tenure:",0,50,5)
st.write(f"Your EMI is {emi} years")


model = ["Gemini","Gemma","Llama3","Phi","OpenAI","Claude","Azure OpenAI"]

choice = st.selectbox("Choose your model:",model)
st.write(f"Your selected model is {choice}:")


uploaded_file = st.file_uploader("Choose a CSV file",type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)