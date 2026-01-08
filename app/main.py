import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# --- DevOps: Instrumentation ---
# Automatically exposes /metrics for Prometheus to scrape
Instrumentator().instrument(app).expose(app)

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Initialize Clients
qdrant = QdrantClient(url=QDRANT_URL)
COLLECTION_NAME = "devops_knowledge"

# --- Data Models ---
class Document(BaseModel):
    text: str
    metadata: dict = {}

class Query(BaseModel):
    question: str

# --- Startup Event: Ensure Vector DB is ready ---
@app.on_event("startup")
async def startup_event():
    # Check if collection exists, if not create it
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")

# --- Endpoints ---

@app.post("/ingest")
async def ingest_document(doc: Document):
    """
    Simulates an ETL pipeline.
    1. Gets embeddings from Ollama (nomic-embed-text is good for this).
    2. Stores them in Qdrant.
    """
    async with httpx.AsyncClient() as client:
        # Get Embedding from Ollama
        response = await client.post(f"{OLLAMA_URL}/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": doc.text
        })
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Embedding failed")
            
        embedding = response.json()["embedding"]
        
        # Store in Qdrant
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=httpx.get("https://www.uuidgenerator.net/api/version4").text.strip(), # Quick hack for UUID
                    vector=embedding,
                    payload={"text": doc.text, **doc.metadata}
                )
            ]
        )
    return {"status": "ingested"}

@app.post("/ask")
async def ask_question(query: Query):
    """
    The RAG Pipeline:
    1. Embed user question.
    2. Search Qdrant for context.
    3. Send Context + Question to Llama3.
    """
    async with httpx.AsyncClient() as client:
        # 1. Embed Question
        emb_res = await client.post(f"{OLLAMA_URL}/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": query.question
        })
        query_vector = emb_res.json()["embedding"]

        # 2. Search Qdrant
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3
        )
        
        context_text = "\n".join([hit.payload["text"] for hit in search_result])

        # 3. Generate Answer
        prompt = f"Context: {context_text}\n\nQuestion: {query.question}\n\nAnswer:"
        
        gen_res = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })
        
        return {
            "answer": gen_res.json()["response"],
            "retrieved_context": [hit.payload for hit in search_result]
        }
