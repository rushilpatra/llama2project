import streamlit as st
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
new_db = FAISS.load_local("medical index", embeddings, allow_dangerous_deserialization=True)
st.title('Medical Chat Bot')
question = st.text_input("Ask me a question")
if question:
    retriever = new_db.as_retriever(search_kwargs={"k": 5})
    context = retriever.get_relevant_documents(question)
    st.write("Relevant documents:")
    st.write(context)
