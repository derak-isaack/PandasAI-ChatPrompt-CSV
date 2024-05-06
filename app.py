import streamlit as st 
import os 
import pandas as pd
from pandasai import Agent
from dotenv import load_dotenv
import matplotlib  
from pandasai.llm import OpenAI
from pandasai import SmartDataframe


matplotlib.use('TkAgg')

st.title("RAG Prompt application to chat with your data")

load_dotenv()
LLM_API_KEY = os.environ['OPENAI_API_KEY']

llm = OpenAI(api_token=LLM_API_KEY)

upload = st.file_uploader("Upload your csv file for analysis", type=['csv'], accept_multiple_files=True)


if upload is not None:
    df = pd.read_csv(upload)
    st.write(df.head(5))
    
    prompt = st.text_input("Enter your prompt")
    
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    
    if st.button('Generate Data'):
        if prompt:
            with st.spinner("Generating data..."):
                st.write(pandas_ai.chat(prompt))
        else:
            st.write("Please enter a prompt:")
