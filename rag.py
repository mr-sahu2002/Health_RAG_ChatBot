import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.llms.base import LLM
from typing import List, Optional, Any, Dict
from llama_cpp import Llama
from pydantic import Field, PrivateAttr


class CustomLLaMA(LLM):
    """Custom LangChain LLM wrapper for llama-cpp"""
    
    model_path: str = Field(..., description="Path to the GGUF model file")
    n_ctx: int = Field(default=2048, description="Context window size")
    n_threads: int = Field(default=12, description="Number of threads to use")
    
    _model: Optional[Llama] = PrivateAttr(default=None)
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 12, **kwargs):
        """Initialize the model"""
        super().__init__(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            **kwargs
        )
        self._model = None
    
    @property
    def model(self) -> Llama:
        """Lazy load the model"""
        if self._model is None:
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=32,
                f16_kv=True,
                vocab_only=False,
                verbose=False
            )
        return self._model
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate text based on the prompt"""
        response = self.model(
            prompt,
            max_tokens=2048,
            temperature=0.7,
            stop=stop or ["</s>", "User:", "\n\n"],
            echo=False
        )
        return response['choices'][0]['text']
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters"""
        return {
            "name": "CustomLLaMA",
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads
        }
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM"""
        return "custom_llama"
    
class RAGApplication:
    def __init__(self, model_path: str):
        """
        Initialize the RAG application with specified models
        
        Args:
            model_path: Path to the GGUF model file
        """
        # Initialize the custom LLM
        self.llm = CustomLLaMA(model_path=model_path)
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}  # Change to 'cuda' if using GPU
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize vector store as None
        self.vector_store = None
        
        # Create custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer: """
        
        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

    def load_documents(self, directory: str):
        """
        Load documents from all text files in the given directory
        
        Args:
            directory: Path to the directory containing text files to load
        """
        documents = []
        # Iterate through all files in the directory
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            # Ensure only text files are processed
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        print(f"Loaded {len(texts)} text chunks into the vector store")
        
    def save_vector_store(self, path: str):
        """
        Save the vector store to disk
        
        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Please load documents first.")
        
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")
    
    def load_vector_store(self, path: str, allow_dangerous_deserialization: bool = False):
        """
        Load a vector store from disk
        
        Args:
            path: Directory path where the vector store is saved
            allow_dangerous_deserialization: Whether to allow loading of pickle files
        """
        self.vector_store = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        print(f"Vector store loaded from {path}")
    
    def query(self, question: str, k: int = 4) -> str:
        """
        Query the RAG system
        
        Args:
            question: The question to ask
            k: Number of relevant documents to retrieve
            
        Returns:
            str: The answer to the question
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Please load documents first.")
        
        # Create the retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT}
        )
        
        # Get the answer
        result = qa_chain.invoke({"query": question})
        return result["result"]


#################### run this file to ingest the new data ####################
if __name__ == "__main__":
    MODEL_PATH = 'model/llama-3.2-1b-instruct-q4_k_m.gguf'
    
    # Initialize the RAG application
    rag = RAGApplication(model_path=MODEL_PATH)
    
    # # create a vector store
    rag.load_documents("data")
    rag.save_vector_store("vector_store")

    # Load existing vector store
    # rag.load_vector_store("vector_store",allow_dangerous_deserialization=True)
    
    # Interactive query loop
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        try:
            answer = rag.query(question)
            print(f"\nQuestion: {question}")
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")