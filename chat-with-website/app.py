
# run with: streamlit run app.py
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage # schema for human and ai messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader # uses BeautifulSoup4
from langchain_community.vectorstores.chroma import Chroma # chroma vector store
from langchain.text_splitter import RecursiveCharacterTextSplitter # split documents into chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from openai import OpenAI
import streamlit as st

import os

load_dotenv()

def get_vectorstore_from_url(url):
    # ingest the webpage into a vectorstore
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(documents = document_chunks, embedding=OpenAIEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    # this function returns a document retriever:
    # the document retriever takes the user query and the chat history and uses it to find relevant documents in the vector store
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    retriever = vector_store.as_retriever() # allows us to retrieve chunks relevant to query
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"), # populates with "chat_history" argument passed to invoke
        ("human", "{input}"), # populates with "input" argument passed to invoke
        ("human", "Given the above conversation, generate a search query to lookup relevant information in the vectorstore."),
    ])

    retriever_chain = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Please answer the user's questions using the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_train = create_retrieval_chain(retriever=retriever_chain, combine_docs_chain=stuff_documents_chain)
    return rag_train

############################################### APP ###############################################

# IMPORTANT: streamlit runs the whole script everytime an input is registered. 
# checking if an object is in session_state prevents it from doing this
if "client" not in st.session_state:
    api_key = os.environ.get('OPENAI_API_KEY')
    st.session_state.client = OpenAI(api_key=api_key)

#page config
st.set_page_config(page_title="Chat With Websites", page_icon="🤖")
st.title("Chat With Websites")

#sidebar
with st.sidebar: # anything in here will appear in the side bar
    st.header("Settings")
    web_url = st.text_input("Website URL")

# main section
if web_url is None or web_url=="":
    st.info("Please enter a web url")
else:
    # save chat history to session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, how can I help you?")
            ]

    #create vector store and save in session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(web_url)
    
    # create RAG chain
    retriever_chain = get_context_retriever_chain(vector_store=st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain=retriever_chain)

    # user query
    user_query = st.chat_input("Enter your message here:")
    if user_query is not None and user_query != "":
        # get response
        response = rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input" : user_query,
        })
        # add to history
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(response['answer']))

        retriever_chain = retriever_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })

    # writes conversation to app
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):        
                st.write(message.content)