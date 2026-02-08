from transformers import pipeline

def load_llm():
    return pipeline(
        task="text-generation",
        model="google/flan-t5-base",
        max_new_tokens=150
    )
