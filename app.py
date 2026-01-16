from fastapi import FastAPI, HTTPException, Form, Query
import chromadb
from chromadb.config import Settings
import requests
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ChromaDB 0.3.x uses Client(Settings(...)) for persistence.
try:
    chroma = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory="./db")
    )
    collection = chroma.get_or_create_collection("docs")
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    raise

# Get Ollama host from environment variable, default to localhost for local development
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_URL = f"http://{OLLAMA_HOST}:11434/api/generate"
logger.info(f"Ollama URL configured: {OLLAMA_URL}")


def process_query(q: str):
    """Helper function to process a query and return the answer."""
    try:
        # Query ChromaDB
        logger.info(f"Querying ChromaDB for: {q}")
        results = collection.query(query_texts=[q], n_results=1)
        
        # Safely extract context
        if results.get("documents") and len(results["documents"]) > 0:
            if len(results["documents"][0]) > 0:
                context = results["documents"][0][0]
            else:
                context = ""
        else:
            context = ""
        
        logger.info(f"Retrieved context length: {len(context)} characters")
        
        # Query Ollama
        logger.info(f"Querying Ollama at {OLLAMA_URL}")
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "tinyllama",
                "prompt": f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:",
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("response", "")
        logger.info(f"Received answer from Ollama")
        
        return {"answer": answer}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama request failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to connect to Ollama at {OLLAMA_URL}: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/")
def root_get(q: str = Query(...)):
    """Root endpoint for GET requests with query parameters."""
    return process_query(q)


@app.post("/")
def root_post(q: str = Form(...)):
    """Root endpoint for POST requests with form data."""
    return process_query(q)


@app.post("/query")
def query(q: str = Form(...)):
    """Query endpoint for POST requests with form data."""
    return process_query(q)
