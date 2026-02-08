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

    # 🔥 STRONG RAG PROMPT (THIS FIXES YOUR BAD ANSWERS)
    prompt = f"""
You are an expert content analyst.

The context below comes from a document that contains tables,
lists, and structured content.

RULES:
- Do NOT repeat the table or raw document text
- Do NOT dump all rows
- Extract ONLY information relevant to the question
- Explain clearly in natural language
- If the question is a single word (like a character name),
  explain what the document says about that topic.

Context:
{context}

Question:
{question}

Answer in one clear, concise paragraph.
"""

    response = llm(prompt)

    return response.strip(), "retrieval"
