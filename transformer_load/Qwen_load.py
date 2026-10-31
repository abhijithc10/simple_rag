from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

load_dotenv()

login(token=os.getenv("HFTOKEN"))

model_name="Qwen/Qwen3.5-0.8B"

model=AutoModelForCausalLM.from_pretrained(model_name,device_map="cuda:0",low_cpu_mem_usage=True,dtype=torch.bfloat16)
tokenizer=AutoTokenizer.from_pretrained(model_name)
print(model.config)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

while True:
    question=input("Prompt:\n")
    if question=="exit":
        break

    tokenized_input=tokenizer(question,return_tensors="pt").to(device)
    model_output=model.generate(tokenized_input["input_ids"],max_length=50,do_sample=True,top_p=0.95,top_k=50)
    decoded_output=tokenizer.decode(model_output[0],skip_special_tokens=True)
    print("\n",decoded_output)