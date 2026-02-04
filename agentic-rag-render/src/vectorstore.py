from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_retriever(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [c.page_content for c in chunks]
    db = Chroma(collection_name="rag_store", embedding_function=embeddings)
    db.add_texts(texts)
    return db.as_retriever(search_kwargs={"k": 3})