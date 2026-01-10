# llmops-lab
demo project


```
docker-compose logs -f rag-api
docker-compose build --no-cache rag-api
OR
docker-compose up --build -d rag-api

docker exec -it llmops-lab-rag-api-1 python -c "import qdrant_client; print(qdrant_client.__file__); print(dir(qdrant_client.QdrantClient))"

fectch model:
    docker exec -it llmops-lab-ollama-1 ollama pull phi3


docker exec -it llmops-lab-rag-api-1 grep "timeout=" /app/main.py

docker logs -f llmops-lab-rag-api-1


curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"text": "Kubernetes is a container orchestration platform.", "metadata": {"source": "docs"}}'
```
