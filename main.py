from fastapi import FastAPI, HTTPException
from rag import RAGApplication
from pydantic import BaseModel

app = FastAPI()

# Path to the model and vector store
MODEL_PATH = 'model/llama-3.2-1b-instruct-q4_k_m.gguf'
VECTOR_STORE_PATH = 'vector_store'

# Initialize the RAG application
rag = RAGApplication(model_path=MODEL_PATH)

# Load the existing vector store
rag.load_vector_store(VECTOR_STORE_PATH, allow_dangerous_deserialization=True)

# Request model for the API endpoint
class QueryRequest(BaseModel):
    question: str

@app.post("/query/")
async def query_rag(request: QueryRequest):
    """
    Endpoint to query the RAG application.

    Args:
        request (QueryRequest): The user's question.
    
    Returns:
        dict: The question and the generated answer.
    """
    try:
        answer = rag.query(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
