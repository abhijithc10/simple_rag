from data_loader.embedder import retriever_load
from transformer_load.Qwen_load import transformer_load
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
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

response=chain.invoke("How many rounds of national allotments will be there ?")
print(response)
