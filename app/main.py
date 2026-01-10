import os
import json
import httpx
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# --- Instrumentation ---
Instrumentator().instrument(app).expose(app)

# --- Config ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION_NAME = "devops_knowledge"

# --- Client Init ---
qdrant = QdrantClient(url=QDRANT_URL)

# --- Models ---
class Document(BaseModel):
    text: str
    metadata: dict = {}

class Query(BaseModel):
    question: str

# --- Startup ---
@app.on_event("startup")
async def startup_event():
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")

# --- Endpoint 1: Ingest ---
@app.post("/ingest")
async def ingest_document(doc: Document):
    async with httpx.AsyncClient() as client:
        # Embed with 60s timeout
        try:
            response = await client.post(f"{OLLAMA_URL}/api/embeddings", json={
                "model": "nomic-embed-text",
                "prompt": doc.text
            }, timeout=60.0)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama embedding failed: {str(e)}")

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Embedding failed")
            
        embedding = response.json()["embedding"]
        point_id = str(uuid.uuid4())
        
        try:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={"text": doc.text, **doc.metadata}
                    )
                ]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Qdrant failed: {str(e)}")
            
    return {"status": "ingested", "id": point_id}

# --- Endpoint 2: Streaming Ask (The Robust Way) ---
# In app/main.py

async def generate_stream(query: str, context: str):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    print(f"DEBUG: Sending prompt to Ollama... (Length: {len(prompt)})") # DevOps Debugging

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("POST", f"{OLLAMA_URL}/api/generate", json={
                "model": "phi3",
                "prompt": prompt,
                "stream": True 
            }, timeout=None) as response:
                
                # FIX: Use aiter_lines() instead of aiter_bytes()
                async for line in response.aiter_lines():
                    if not line: continue # Skip empty keep-alive lines
                    
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except Exception as e:
                        print(f"JSON Parse Error: {e}") # Log it so we see it
                        continue
                        
        except Exception as e:
            print(f"Stream Connection Error: {e}")
            yield f"Error: {str(e)}"

@app.post("/ask_stream")
async def ask_question_stream(query: Query):
    async with httpx.AsyncClient() as client:
        # 1. Embed
        emb_res = await client.post(f"{OLLAMA_URL}/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": query.question
        }, timeout=60.0)
        query_vector = emb_res.json()["embedding"]

        # 2. Search
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        ).points
        context_text = "\n".join([hit.payload["text"] for hit in search_result])

    # 3. Stream Response
    return StreamingResponse(
        generate_stream(query.question, context_text), 
        media_type="text/event-stream"
    )

# --- Endpoint 3: Blocking Ask (Backup) ---
@app.post("/ask")
async def ask_question(query: Query):
    async with httpx.AsyncClient() as client:
        # 1. Embed
        emb_res = await client.post(f"{OLLAMA_URL}/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": query.question
        }, timeout=60.0)
        query_vector = emb_res.json()["embedding"]

        # 2. Search
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        ).points
        context_text = "\n".join([hit.payload["text"] for hit in search_result])

        # 3. Generate (With 120s Timeout)
        prompt = f"Context: {context_text}\n\nQuestion: {query.question}\n\nAnswer:"
        
        gen_res = await client.post(f"{OLLAMA_URL}/api/generate", json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        }, timeout=120.0)
        
        return {
            "answer": gen_res.json()["response"],
            "retrieved_context": [hit.payload for hit in search_result]
        }