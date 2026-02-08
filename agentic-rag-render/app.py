import streamlit as st
from src.rag_pipeline import rag_answer, load_and_index_pdfs
import os

st.set_page_config(page_title="Agentic RAG", layout="centered")

st.title("🤖 Agentic RAG System")
st.write("Upload PDFs and ask questions.")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data", exist_ok=True)

    for file in uploaded_files:
        with open(f"data/{file.name}", "wb") as f:
            f.write(file.read())

    with st.spinner("Indexing PDFs..."):
        load_and_index_pdfs("data")

    st.success("PDFs indexed successfully!")

question = st.text_input("Ask a question")

if question:
    with st.spinner("Thinking..."):
        answer, action = rag_answer(question)

    st.markdown(f"**Agent decision:** `{action}`")
    st.markdown("### Answer")
    st.write(answer)
