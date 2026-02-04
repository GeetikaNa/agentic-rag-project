from src.agent import agent_controller
from src.loader import load_docs
from src.chunker import chunk_docs
from src.vectorstore import create_retriever
from src.llm import load_llm

DATA_PATH = "data"

docs = load_docs(DATA_PATH)
chunks = chunk_docs(docs)
retriever = create_retriever(chunks)
llm = load_llm()

def rag_answer(query):
    action = agent_controller(query)

    if action == "search":
        results = retriever.invoke(query)
        context = "\n".join([r.page_content for r in results])
        prompt = f"Use this context:\n{context}\n\nAnswer:\n{query}"
    else:
        prompt = query

    return llm(prompt)[0]["generated_text"]