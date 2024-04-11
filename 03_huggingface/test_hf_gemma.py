from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline

load_dotenv()
model_id = "google/gemma-2b-it"
input_text = "Write me a poem about Machine Learning."
USE_PIPELINE = True

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir ='models/', return_tensorts='pt', add_special_tokens=False)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir ='models/', device_map='cuda', torch_dtype = torch.float16)
    
if not USE_PIPELINE:
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens = 200)
    print(tokenizer.decode(outputs[0]))

if USE_PIPELINE:
    print('Using Pipeline')
    messages = [
    {"role": "user", "content": f"{input_text}"},
    ]
    #pipe = pipeline(task='text-generation', model=model, tokenizer=tokenizer)
    pipe = pipeline(task='text-generation', model = model, tokenizer=tokenizer, max_new_tokens=200)

    outputs = pipe(messages)
    print(outputs[0]['generated_text'][-1]['content'])