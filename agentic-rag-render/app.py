from flask import Flask, request, jsonify
from src.rag_pipeline import rag_answer

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"status": "Agentic RAG running"}

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query", "")
    answer = rag_answer(query)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run()