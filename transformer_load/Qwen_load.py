from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
from huggingface_hub import login
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
import os
import torch

def transformer_load(): #initialize and load transformer model
    load_dotenv()

    login(token=os.getenv("HFTOKEN"))

    model_name="Qwen/Qwen3.5-0.8B"

    model=AutoModelForCausalLM.from_pretrained(model_name,device_map="cuda:0",low_cpu_mem_usage=True,dtype=torch.bfloat16)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=1024,top_k=100,top_p=0.95,temperature=0.2,
                return_full_text=False,repetition_penalty=1.1)
    llm=HuggingFacePipeline(pipeline=pipe)
    chatllm=ChatHuggingFace(llm=llm,tokenizer=tokenizer)
    return chatllm

