from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
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

# Load the Hugging Face model (replace with your desired model)
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 50},
    device=0)  # -1 for CPU

output_parser = StrOutputParser()
chain = prompt|llm|output_parser

## streamlit
st.title('Langchain Chatbot using Hugging Face')
input_text = st.text_input("Please enter query")

if input_text:
    st.write(chain.invoke({'question':input_text}))