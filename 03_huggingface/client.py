## streamlit
import requests
import streamlit as st


def get_response(input_text):
    response=requests.post(
        "http://localhost:8000/llm/invoke",
        json = {'input':{'question':input_text}})
    return response.json()['output'].split('<start_of_turn>model\n')[-1]

st.title('Langchain Chatbot using Hugging Face')
input_text = st.text_input("Please enter query")

if input_text:
    st.write(get_response(input_text))