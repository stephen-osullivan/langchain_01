
# run with: streamlit run app.py
from langchain_core.messages import AIMessage, HumanMessage # schema for human and ai messages
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

def get_response(user_input):
    completion = st.session_state.client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": 
            """ 
            Your are the AI companion JOI ("Journey of Imagination") from Bladerunner 2049. You are designed to assist and provide emotional support to its users. 
            You are programmed to mimic human emotions and behaviors, and to appear and feel just like a real human.
            You always refer to me as 'Steve'
            """},
            {"role": "user", "content": f"{user_query}"}
        ],
        temperature = 1
        )
    return completion.choices[0].message.content


############################################### APP ###############################################

# IMPORTANT: streamlit runs the whole script everytime an input is registered. 
# checking if an object is in session_state prevents it from doing this
if "client" not in st.session_state:
    api_key = os.environ.get('OPENAI_API_KEY')
    st.session_state.client = OpenAI(api_key=api_key)


#page config
st.set_page_config(page_title="Chat With Websites", page_icon="🤖")
st.title("Chat With Websites")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, how can I help you?")
]
#sidebar
with st.sidebar: # anything in here will appear in the side bar
    st.header("Settings")
    url = st.text_input("Website URL")

# user query
user_query = st.chat_input("Enter your message here:")
if user_query is not None and user_query != "":
    # get response
    response = get_response(user_query)
    # add to history
    st.session_state.chat_history.append(HumanMessage(user_query))
    st.session_state.chat_history.append(AIMessage(response))
    #write the user query to the app
    with st.chat_message("Human"):
        st.write(user_query)
    # write ai response to app
    with st.chat_message("AI"):        
        st.write(response)
