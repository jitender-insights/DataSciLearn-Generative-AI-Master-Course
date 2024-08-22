"""
Streamlit is an open-source Python library that simplifies the process of creating and sharing data applications. 
It allows data scientists, machine learning engineers, and developers to quickly build interactive web applications 
directly from Python scripts. 
With Streamlit, you can create data-driven applications with minimal effort, 
as it handles much of the web development work behind the scenes.


"""

import streamlit as st 
import pandas as pd
import numpy as np



#Title of the application

st.title("Hello Data Science Learner")


#Display a simple text
st.write("Welcome to our Youtube Channel @DataSciLearn")


#Create a DataFrame

df = pd.DataFrame(
    {
        'paid LLm model':["claude","gemini","Open AI"],
        "free LLm model":["llama3","gemma","phi"]
    }
)

#display the dataframe
st.write("Here is the dataframe")
st.write(df)


#Create a line chart
chart_data = pd.DataFrame(
    np.random.randn(20,4), columns=['a','b','c','d']
)

st.line_chart(chart_data)

