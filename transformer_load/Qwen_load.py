from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline
from huggingface_hub import login
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from langchain.messages import HumanMessage,SystemMessage
import os
import torch

load_dotenv()

login(token=os.getenv("HFTOKEN"))

model_name="Qwen/Qwen3.5-0.8B"

model=AutoModelForCausalLM.from_pretrained(model_name,device_map="cuda:0",low_cpu_mem_usage=True,dtype=torch.bfloat16)
tokenizer=AutoTokenizer.from_pretrained(model_name)
pipe=pipeline("text-generation",model=model,tokenizer=tokenizer,max_new_tokens=1024,top_k=100,top_p=0.95,temperature=0.2,
              return_full_text=False,repetition_penalty=1.1)
llm=HuggingFacePipeline(pipeline=pipe)
chatllm=ChatHuggingFace(llm=llm,tokenizer=tokenizer)

sys_message=SystemMessage("You are a helpful AI who answers the questions asked of you.answer the question the human message directly")


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

while True:
    question=input("Prompt:\n")
    if question=="exit":
        break
    hum_message=HumanMessage(question)
    message=[sys_message,hum_message]

    output=chatllm.invoke(message)

    #tokenized_input=tokenizer(question,return_tensors="pt").to(device)
    #model_output=model.generate(tokenized_input["input_ids"],max_length=50,do_sample=True,top_p=0.95,top_k=50)
    #decoded_output=tokenizer.decode(model_output[0],skip_special_tokens=True)
    print("\n",output.content)

