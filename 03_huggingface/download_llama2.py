from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

load_dotenv()
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, return_tensorts='pt', add_special_tokens=False)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', torch_dtype = torch.float16)
