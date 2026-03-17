# "DocuMind" — Multi-format RAG with Smart Retrieval

## What it does:

A CLI + Streamlit web app where users upload PDFs, Word docs, or paste URLs. The system chunks, embeds, and indexes them. Users chat with their documents. Supports multiple documents simultaneously with source citation in every answer.
What makes it non-trivial:

## Implement EnsembleRetriever (BM25 + dense vector hybrid with RRF fusion)

Add ContextualCompressionRetriever with a cross-encoder re-ranker
Persist vector store to disk (ChromaDB) so documents survive restarts
Show retrieved source chunks alongside every answer with relevance scores
Add a confidence threshold — if top chunk score is below X, say "I don't know" instead of hallucinating

## Tech stack:

LangChain, ChromaDB, OpenAI embeddings, Streamlit, sentence-transformers (re-ranker)

## GitHub showcase value:

Demonstrates RAG fundamentals + retrieval engineering + honest uncertainty handling. Interviewers love the hybrid retrieval + re-ranking combo.
