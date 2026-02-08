from src.loader import load_docs
from src.vectorstore import create_retriever
from src.llm import load_llm

_llm = None
_retriever = None


def load_and_index_pdfs(data_path: str):
    global _retriever
    docs = load_docs(data_path)
    _retriever = create_retriever(docs)
    return _retriever


def rag_answer(question: str):
    global _llm, _retriever

    if _llm is None:
        _llm = load_llm()

    if _retriever is None:
        _retriever = load_and_index_pdfs("data")

    docs = _retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
Answer the question clearly using the context.

Context:
{context}

Question:
{question}
"""

    result = _llm(prompt)

    # ✅ CORRECT extraction for FLAN-T5
    if isinstance(result, list):
        answer = result[0]["generated_text"].strip()
    else:
        answer = str(result).strip()

    # ✅ Safety fallback
    if not answer:
        answer = "No relevant information found in the uploaded documents."

    return answer, "retrieval"
