from src.loader import load_docs
from src.vectorstore import create_retriever
from src.llm import load_llm

def load_and_index_pdfs(data_path: str):
    docs = load_docs(data_path)
    retriever = create_retriever(docs)
    return retriever


def rag_answer(question: str):
    llm = load_llm()

    retriever = load_and_index_pdfs("data")

    docs = retriever.get_relevant_documents(question)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm(prompt)
    return response, "retrieval"
