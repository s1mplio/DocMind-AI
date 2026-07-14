from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
EmbeddingModel="sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_PATH="vectorstore"
token=os.getenv("HF_TOKEN")

#load embedding model
embedding_model=HuggingFaceEmbeddings(model_name=EmbeddingModel)

#load FAISS index


#create retriever 



#load qwen

llm=ChatGoogleGenerativeAI(model="gemini-3-flash-preview", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

prompt=ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
""")

def ask(question):
    vector_store=FAISS.load_local(VECTOR_DB_PATH,
                              embedding_model,
                              allow_dangerous_deserialization=True
                              )
    retriever=vector_store.as_retriever(search_kwargs={"k":3})
    docs=retriever.invoke(question)
    context="\n\n".join([doc.page_content for doc in docs])
    chain=prompt | llm  | StrOutputParser()
    response=chain.invoke({
        "context":context,
        "question":question
    })
    return response

if __name__=="__main__":
    while True:
        question=input("Ask:")
        if question.lower()=="exit":
            break;
        
        answer=ask(question)

        
        print(f"Answer:{answer}")

                        
