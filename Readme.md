# Health ChatBot

This project leverages various AI technologies and tools to scrape, store, and process medical data from WebMD and provide fast access to relevant information via an API.

## Tech Stack

- **LLM (1-1.5B parameters)**: [Llama-3.2-1B-Instruct-Q4_K_M-GGUF](https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF)
- **Framework**: [Langchain](https://langchain.com/)
- **VectorDB**: [FAISS](https://github.com/facebookresearch/faiss)
- **Embedding Model**: [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **Web Scraping**: [Firecrawl](https://llmstxt.firecrawl.dev/) & [Firecrawl (official)](https://www.firecrawl.dev/)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Caching**: [Redis](https://redis.io/)
- **Containerization**: [Docker](https://www.docker.com/)

## Data Sources

- **WebMD**: [https://www.webmd.com/](https://www.webmd.com/)

## How to Run

1. Clone the repository to your local machine.
   
   ```bash
   git clone https://github.com/mr-sahu2002/Health_RAG_ChatBot.git
2. After cloning the repo open the terminal on the same folder 

3. create the folder model in root directory

4. download the model from huggingface 
https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF

5. place the .gguf file in model directory

6. install docker on you machine 

7. start the docker desktop

8. Run this command in the root directory

   ```bash
   docker-compose up --build 
9. Aplication will start and you can test the api 
http://localhost:8000/docs

10. All done!