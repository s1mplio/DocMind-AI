import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


load_dotenv()

token=os.getenv("HF_TOKEN")

VECTOR_STORE_PATH="vectorstore"

def load_document(pdf_path):
    
        loader=PyPDFLoader(pdf_path)
        documents=loader.load()
        return documents

def split_documents(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks=text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    embeddings_huggingface=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",token=token)
    vector_store=FAISS.from_documents(documents=chunks,embedding=embeddings_huggingface)
    os.makedirs(VECTOR_STORE_PATH,exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    print("Successfully created vector store and saved to local path")


def ingest(pdf_path):
    documents=load_document(pdf_path) 
    chunks=split_documents(documents)
    create_vector_store(chunks)
            


if __name__=="__main__":
    ingest("uploads/sample.pdf")
