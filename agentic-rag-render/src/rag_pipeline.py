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

    # 🔒 TIGHT PROMPT (NO ECHO)
    prompt = f"""
Context information:
{context}

Question:
{question}

Answer (ONLY the answer, no context, no rules, no lists):
"""

    result = _llm(prompt)

    # ✅ HuggingFace pipeline output handling
    if isinstance(result, list) and "generated_text" in result[0]:
        answer = result[0]["generated_text"]
    else:
        answer = str(result)

    # 🧹 HARD CLEANUP (IMPORTANT)
    answer = answer.replace(prompt, "").strip()

    # 🧠 Keyword guard
    if len(question.split()) <= 2 and len(answer) > 600:
        answer = answer.split(".")[0] + "."

    return answer, "retrieval"
