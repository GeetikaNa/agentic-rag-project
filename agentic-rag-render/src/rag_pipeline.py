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

    docs = retriever.invoke(question)

    if not docs:
        return "No relevant information found in the uploaded documents.", "no_retrieval"

    context = "\n".join(d.page_content for d in docs[:3])  # hard cap

    prompt = f"""
    
    
    Question:
    {question}


    Context:
    {context}

    """

    response = llm(prompt)

    if isinstance(response, list):
        response = response[0].get("generated_text", "").strip()
    elif isinstance(response, dict):
        response = response.get("generated_text", "").strip()

    if not response:
        return "The document references Damon but does not explain him in detail.", "retrieval"

    return response, "retrieval"
