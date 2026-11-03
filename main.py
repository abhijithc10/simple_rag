from data_loader.embedder import retriever_load
from transformer_load.Qwen_load import transformer_load
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from pydantic import BaseModel

def format_docs(docs): # format the retriver output to add metadata
    final_formatted_chunks=[]
    for doc in docs:
        source=doc.metadata["source"]
        page_num=doc.metadata["page"]
        chunk=f"[Source : {source} , page number : {page_num}] \n {doc.page_content}]"
        final_formatted_chunks.append(chunk)
    return "\n\n--\n\n".join(final_formatted_chunks)


retriever=retriever_load() # loads the chroma db retriever
model=transformer_load() #loads the llm model

template=ChatPromptTemplate.from_messages([
    ("system","you are an assistant helping to answer questions about neet medical pg seat allotment. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.\n\nContext:\n{context}"),
    ("human","{question}")
])

chain=(
    {"context":retriever | format_docs , "question":RunnablePassthrough()}
    |template
    |model
    |StrOutputParser()

)


class request(BaseModel):
    question:str


app=FastAPI()

@app.get('/')
def root():
    return "works!"

@app.post('/chat/')
def chat(request:request):
    response=chain.invoke(request.question)
    return {"response":response}
