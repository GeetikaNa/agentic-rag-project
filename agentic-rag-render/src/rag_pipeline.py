from src.loader import load_docs
from src.vectorstore import create_retriever
from src.llm import load_llm

# Load & index PDFs only once (important for Streamlit)
retriever = None
llm = None


def load_and_index_pdfs(data_path: str):
    global retriever

    if retriever is None:
        docs = load_docs(data_path)
        retriever = create_retriever(docs)

    return retriever


def rag_answer(question: str):
    global llm

    if llm is None:
        llm = load_llm()

    retriever = load_and_index_pdfs("data")

    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are an expert academic assistant.

Rules:
- Do NOT dump raw text or tables
- Summarize clearly
- Answer only what the question asks
- If the query is a keyword, explain it using the document

Context:
{context}

Question:
{question}

Answer in one clear paragraph.
"""

    response = llm(prompt)

    # 🔥 HANDLE HUGGINGFACE PIPELINE OUTPUT
    if isinstance(response, list) and "generated_text" in response[0]:
        answer = response[0]["generated_text"]
    else:
        answer = str(response)

    return answer.strip(), "retrieval"
