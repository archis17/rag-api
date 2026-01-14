from fastapi import FastAPI
import chromadb
from chromadb.config import Settings
import requests

app = FastAPI()

# ChromaDB 0.3.x uses Client(Settings(...)) for persistence.
chroma = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory="./db")
)
collection = chroma.get_or_create_collection("docs")


@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:",
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    return {"answer": data.get("response", "")}
