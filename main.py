from fastapi import FastAPI, HTTPException
from rag import RAGApplication
from pydantic import BaseModel
import redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Path to the model and vector store
MODEL_PATH = 'model/llama-3.2-1b-instruct-q4_k_m.gguf'
VECTOR_STORE_PATH = 'vector_store'

# Initialize the RAG application
rag = RAGApplication(model_path=MODEL_PATH)

# Load the existing vector store
rag.load_vector_store(VECTOR_STORE_PATH, allow_dangerous_deserialization=True)

# Initialize Redis client
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Load a pre-trained embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Request model for the API endpoint
class QueryRequest(BaseModel):
    question: str

def get_question_embedding(question: str) -> np.ndarray:
    """Generate an embedding for a given question."""
    return embedding_model.encode(question)

def find_similar_question(question_embedding: np.ndarray, threshold: float = 0.8):
    """Search Redis for the most similar question based on cosine similarity."""
    keys = redis_client.keys("question:*")
    for key in keys:
        cached_embedding = np.frombuffer(redis_client.hget(key, "embedding"), dtype=np.float32)
        similarity = np.dot(cached_embedding, question_embedding) / (
            np.linalg.norm(cached_embedding) * np.linalg.norm(question_embedding)
        )
        if similarity > threshold:
            return key, redis_client.hget(key, "answer").decode("utf-8")
    return None, None

@app.post("/query/")
async def query_rag(request: QueryRequest):
    """
    REST API Endpoint to query the RAG application with similarity-based caching.

    Args:
        request (QueryRequest): The user's question.
    
    Returns:
        dict: The question and the generated answer.
    """
    try:
        # Generate the embedding for the question
        question_embedding = get_question_embedding(request.question)

        # Check for a similar question in Redis
        cache_key, cached_answer = find_similar_question(question_embedding)
        if cached_answer:
            return {"question": request.question, "answer": cached_answer}

        # If no similar question, query the RAG model
        answer = rag.query(request.question)

        # Cache the result in Redis
        question_key = f"question:{hash(request.question)}"
        redis_client.hset(
            question_key,
            mapping={
                "embedding": question_embedding.tobytes(),
                "answer": json.dumps(answer),
            }
        )
        redis_client.expire(question_key, 86400)  # Cache for 1 day

        return {"question": request.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
