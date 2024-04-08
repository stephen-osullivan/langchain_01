
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import uvicorn
import os

"""
a simple fast api app that allows the user to switch between openai and ollama
run with: $ python app.py

"""
load_dotenv()

### Define the chains
prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "You are a useful and attentive assitant. Please respond to user queries."),
        ("human", "Question:{question}"),
    ]
)
# models
openai_llm = ChatOpenAI()
ollama_llm = Ollama(model="llama2")
# output parser
output_parser = StrOutputParser()
# chains
openai_chain = prompt | openai_llm | output_parser
ollama_chain = prompt | ollama_llm | output_parser

### FAST API APP
app = FastAPI(
    title='Langchain Server',
    version='1.0',
    description='A Simple API Server',
)
# LANGSERVE routes
add_routes(
    app,
    openai_chain,
    path = "/openai",
)
add_routes(
    app,
    ollama_chain,
    path = "/ollama",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)