# Core

poetry add fastapi uvicorn[standard]

## LangChain ecosystem

poetry add langchain langchain-core langchain-community langchain-ollama
poetry add langgraph langsmith

### Document loaders

poetry add pypdf python-docx unstructured[pdf]

#### Vector store + embeddings

poetry add chromadb ollama tiktoken

## Text splitting

poetry add langchain-text-splitters

# Retrieval

poetry add rank-bm25 cohere

## Utilities

poetry add python-dotenv pydantic-settings httpx

### Dev dependencies (only installed in dev)

poetry add --group dev pytest pytest-asyncio black ruff mypy ipython
