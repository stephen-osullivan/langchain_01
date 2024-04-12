from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.pipelines import pipeline
import uvicorn

import argparse
import os
import sys

load_dotenv()

print('Utilizing GPU? :', torch.cuda.is_available())

parser = argparse.ArgumentParser(description='Load LLM as an endpoint')
parser.add_argument('--model', type=str, default='llama2', help='llama2 or gemma')
parser.add_argument('--quantize', type=bool, default='False', help='llama2 or gemma')

args = parser.parse_args()
model_name = args.model
quantize = args.quantize
print('Using', model_name)
print('Quantizing?:', quantize)

if model_name == 'gemma':
    model_id = 'google/gemma-2b-it'
    ## prompt template specifically for GEMMA
    prompt = PromptTemplate.from_template(
    """
    <bos><start_of_turn>user
    {question}<end_of_turn>
    <start_of_turn>model
    """
    )
elif model_name =='llama2':
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "You are a useful and attentive assitant. Please respond to user queries. Please keep your answer very concise."),
            ("human", "Question:{question}"),
        ]
    )
else:
    print('Invalid Model Name')
    sys.exit()



tokenizer = AutoTokenizer.from_pretrained(model_id, return_tensorts='pt', add_special_tokens=False)
print('Loaded Tokenizer')

if quantize:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', quantization_config=bnb_config)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', torch_dtype = torch.float16)
print('Loaded Model')

# Load into pipelines

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200,
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