from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import uvicorn

import os

load_dotenv()
## prompt template specifically for GEMMA
prompt = PromptTemplate.from_template(
"""
<bos><start_of_turn>user
{question}<end_of_turn>
<start_of_turn>model
"""
)

    
model_id = 'google/gemma-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir ='models/')
print('Loaded Tokenizer')
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir ='models/', torch_dtype = torch.float16)
print('Loaded Model')

# Load the Hugging Face model (replace with your desired model)

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, device='cuda',
)
llm = HuggingFacePipeline(pipeline=pipe)


chain = prompt|llm
### FAST API APP
app = FastAPI(
    title='Langchain Server',
    version='1.0',
    description='A Simple API Server',
)

### LANGSERVE route
add_routes(
    app,
    chain,
    path = "/llm",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)