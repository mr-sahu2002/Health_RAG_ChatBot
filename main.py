from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from rag import RAGApplication
from pydantic import BaseModel
from redis.asyncio import ConnectionPool, Redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

# Define the request model
class QueryRequest(BaseModel):
    question: str

# Constants
MODEL_PATH = 'model/llama-3.2-1b-instruct-q4_k_m.gguf'
VECTOR_STORE_PATH = 'vector_store'

# Global variables to hold our resources
thread_pool = None
redis_pool = None
embedding_model = None
rag_pool = None

class RAGPool:
    def __init__(self, pool_size: int = 5):
        self.pool = [
            RAGApplication(model_path=MODEL_PATH)
            for _ in range(pool_size)
        ]
        for rag in self.pool:
            rag.load_vector_store(VECTOR_STORE_PATH, allow_dangerous_deserialization=True)
        self._lock = asyncio.Lock()
        self._current = 0

    async def get_rag(self):
        async with self._lock:
            rag = self.pool[self._current]
            self._current = (self._current + 1) % len(self.pool)
            return rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources
    global thread_pool, redis_pool, embedding_model, rag_pool
    
    print("Starting up...")
    thread_pool = ThreadPoolExecutor(max_workers=5)
    redis_pool = ConnectionPool(
        host='redis',
        port=6379,
        db=0,
        max_connections=10
    )
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    rag_pool = RAGPool(pool_size=5)
    
    yield
    
    # Cleanup
    print("Shutting down...")
    thread_pool.shutdown(wait=True)
    await redis_pool.disconnect()

# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG API",
    description="RAG Application with Redis Caching",
    version="1.0.0",
    lifespan=lifespan
)

async def get_question_embedding(question: str) -> np.ndarray:
    """Generate an embedding for a given question asynchronously."""
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool,
        lambda: embedding_model.encode(question)
    )

async def find_similar_question(
    redis_client: Redis,
    question_embedding: np.ndarray,
    threshold: float = 0.8
) -> tuple[Optional[bytes], Optional[str]]:
    """Search Redis for the most similar question based on cosine similarity."""
    keys = await redis_client.keys("question:*")
    for key in keys:
        cached_embedding_bytes = await redis_client.hget(key, "embedding")
        if cached_embedding_bytes:
            cached_embedding = np.frombuffer(cached_embedding_bytes, dtype=np.float32)
            similarity = np.dot(cached_embedding, question_embedding) / (
                np.linalg.norm(cached_embedding) * np.linalg.norm(question_embedding)
            )
            if similarity > threshold:
                answer = await redis_client.hget(key, "answer")
                if answer:
                    return key, answer.decode("utf-8")
    return None, None

@app.post("/query/",
         response_model=dict,
         summary="Query the RAG system",
         description="Submit a question to the RAG system with Redis caching")
async def query_rag(request: QueryRequest):
    """Query endpoint for the RAG application with similarity-based caching."""
    async with Redis(connection_pool=redis_pool) as redis_client:
        try:
            question_embedding = await get_question_embedding(request.question)

            cache_key, cached_answer = await find_similar_question(
                redis_client,
                question_embedding
            )
            
            if cached_answer:
                return {"question": request.question, "answer": cached_answer}

            rag = await rag_pool.get_rag()
            
            answer = await asyncio.get_event_loop().run_in_executor(
                thread_pool,
                lambda: rag.query(request.question)
            )

            question_key = f"question:{hash(request.question)}"
            await redis_client.hset(
                question_key,
                mapping={
                    "embedding": question_embedding.tobytes(),
                    "answer": json.dumps(answer),
                }
            )
            await redis_client.expire(question_key, 86400)  # Cache for 1 day
            
            return {"question": request.question, "answer": answer}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health",
        response_model=dict,
        summary="Health check",
        description="Check if the API is running")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}