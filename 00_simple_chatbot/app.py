from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

import os

load_dotenv()

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a useful and attentive assitant. Please respond to user queries."),
        ("human", "Question:{question}"),
    ]
)

## model
llm = ChatOpenAI(model='gpt-3.5-turbo')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

## streamlit
st.title('Langchain Chatbot using OPENAI')
input_text = st.text_input("Please enter query")

if input_text:
    st.write(chain.invoke({'question':input_text}))