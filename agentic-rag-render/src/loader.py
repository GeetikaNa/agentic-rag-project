import os
from langchain_community.document_loaders import PyPDFLoader

def load_docs(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            docs.extend(loader.load())
    return docs