from langchain_community.llms import ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

import os

## prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a useful and attentive assitant. Please respond to user queries."),
        ("human", "Question:{question}"),
    ]
)

## model
llm = ollama.Ollama(model='llama2')
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

## streamlit
st.title('Langchain Chatbot using OLLAMA')
input_text = st.text_input("Please enter query")

if input_text:
    st.write(chain.invoke({'question':input_text}))