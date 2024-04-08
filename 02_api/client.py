import requests
import streamlit as st

"""
streamlit app to query to fastapi app defined in app.py
ensure the app in app.py is running
ollama must all be running
"""
def get_openai_response(input_text):
    response=requests.post(
        "http://localhost:8000/openai/invoke",
        json = {'input':{'question':input_text}})
    return response.json()['output']

def get_ollama_response(input_text):
    response=requests.post(
        "http://localhost:8000/ollama/invoke",
        json = {'input':{'question':input_text}})
    return response.json()['output']

### Streamlit app
st.title('LLM API Web APP')
llm = st.selectbox('llm', options = ['openai', 'ollama'])
input_text = st.text_input('query')
if input_text:
    if llm == 'openai':
        output_text = get_openai_response(input_text)
    if llm == 'ollama':
        output_text = get_ollama_response(input_text)
    st.write(output_text)