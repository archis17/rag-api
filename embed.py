import chromadb
from chromadb.config import Settings

# ChromaDB 0.3.x uses `Client(Settings(...))` for persistence.
client = chromadb.Client(
    Settings(chroma_db_impl="duckdb+parquet", persist_directory="./db")
)
collection = client.get_or_create_collection("docs")

with open("k8s.txt", "r") as f:
    text = f.read()

if hasattr(collection, "upsert"):
    collection.upsert(documents=[text], ids=["k8s"])
else:
    # Fallback for older Chroma versions.
    try:
        collection.delete(ids=["k8s"])
    except Exception:
        pass
    collection.add(documents=[text], ids=["k8s"])
client.persist()

print("Embedding stored in Chroma")