import os
import re
import pymupdf
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model="Qwen/Qwen3-Embedding-0.6B"

pdf_path="data_loader/pdfdrop"
database_path="data_loader/chroma_db"
pdf_drop_files=os.listdir(pdf_path)

#to remove identifier files since this is running in wsl
reg=re.compile(r'\.(Identifier)$')
pdf_drop_files=[s for s in pdf_drop_files if reg.search(s)==None]

def create_database(list_pdf): #takes a list of pdf names&embedding_model,chunks it and saves the vector database onto disk and returns the retriever 
    embedding=HuggingFaceEmbeddings(model_name=embedding_model,model_kwargs={"device":"cuda:0"})
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,separators=["\n\n", "\n", " ", ""])
    doc_num=0
    for doc in list_pdf:
        pdf=pymupdf.open(f'{pdf_path}/{doc}')
        text=[]
        for pagenum in range(len(pdf)):
            page=pdf.load_page(pagenum)
            page_text=page.get_text()
            text.append(Document(page_content=page_text,metadata={"source":doc,"page":pagenum+1}))
        chunks=text_splitter.split_documents(documents=text)
        if doc_num==0:
            vector_store=Chroma.from_documents(documents=chunks,embedding=embedding,persist_directory=database_path)
            print(f"created chroma_db \n added {doc}")
        else:
            vector_store.add_documents(chunks)
            print(f"added {doc}")
        doc_num=doc_num+1
    return vector_store.as_retriever(search_kwargs={"k":4})

def load_database(): # loads the existing chroma database
    embedding=HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store=Chroma(persist_directory=database_path,embedding_function=embedding)
    return vector_store.as_retriever(search_kwargs={"k":4})

def retriever_load():# create chroma_db at first and load it subsequent times
    if os.listdir(database_path)==[]:
        return create_database(pdf_drop_files)
    else:
        return load_database()











