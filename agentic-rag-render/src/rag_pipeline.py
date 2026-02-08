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

    if not docs:
        return "No relevant information found in the document.", "retrieval"

    context = "\n".join(d.page_content for d in docs)

    # 🔑 FLAN-T5-OPTIMIZED PROMPT
    prompt = f"""
Context:
{context}

Question:
What does the document say about {question}?

Answer in ONE clear paragraph. Do not repeat the context.
"""

    result = _llm(prompt)

    if isinstance(result, list):
        answer = result[0]["generated_text"].strip()
    else:
        answer = str(result).strip()

    # 🧹 Remove prompt echo if it appears
    if "Context:" in answer:
        answer = answer.split("Answer")[-1].strip()

    if not answer:
        answer = "The document does not contain a clear explanation for this query."

    return answer, "retrieval"
