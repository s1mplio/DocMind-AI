import streamlit as st
import os
from ingest import ingest
from rag import ask

st.set_page_config(
  page_title="DocMind-AI",
  page_icon="📄",
  layout="wide"
)

#session state initialization

if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready=False

if "messages" not in st.session_state:
    st.session_state.messages=[]

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path=None







st.title("DocMind-AI: Your Document Assistant")
st.markdown("Upload a PDF document and ask questions about its content.")

#create upload folder

upload_folder="uploads"
os.makedirs(upload_folder,exist_ok=True)

#upload pdf
with st.sidebar:
    uploaded_file=st.file_uploader("Choose a pdf file",type=["pdf"])
    build=st.button("Build Knowledge Base",
                    disabled=uploaded_file is None)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages=[]
        st.rerun()


#build knowledge base

if uploaded_file is not None:
    pdf_path=os.path.join(upload_folder,uploaded_file.name)
    
    if st.session_state.pdf_path!=pdf_path:
        with open(pdf_path,"wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.pdf_path=pdf_path
        st.session_state.knowledge_base_ready=False
        st.session_state.messages=[]
        st.success(f"saved file:{uploaded_file.name}")
    
    if build:
        with st.spinner("Building knowledge base.."):
            ingest(pdf_path)
        st.session_state.knowledge_base_ready=True
        st.success("Knowledge base built successfully!")
    
#ask questions

st.divider()

questions=st.chat_input("Ask a question about the document:")

if questions:

    if not st.session_state.knowledge_base_ready:
        st.warning("please upload a document and build the knowledge base first.")
    elif questions.strip()=="":
        st.warning("Please ask a question.")
    else:
        with st.spinner("Getting Answer.."):
            answer=ask(questions)
            st.session_state.messages.append({"role":"user","content":questions})
            st.session_state.messages.append({"role":"assistant","content":answer})
        st.rerun()
        
for message in st.session_state.messages:
    
       with st.chat_message(message["role"]):
            st.write(message["content"])




