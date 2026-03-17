# DocuMind AI — Technical Architecture Guide

## Overview

DocuMind is an AI-powered document intelligence platform built on a Retrieval-Augmented Generation (RAG) architecture. It allows users to upload documents and ask natural language questions, receiving accurate answers grounded in the document content.

---

## System Architecture

DocuMind uses a **three-layer retrieval pipeline**:

### Layer 1: Hybrid Retrieval

| Method | Technology | Best For |
|--------|-----------|----------|
| Dense (Semantic) | ChromaDB + nomic-embed-text | Conceptual queries, paraphrased questions |
| Sparse (Keyword) | BM25 | Exact terms, clause numbers, IDs |
| Fusion | Reciprocal Rank Fusion (RRF) | Combines both signals |

Weights are configured at `0.6 dense / 0.4 sparse` — tunable per corpus type.

### Layer 2: Re-Ranking

- Model: `ms-marco-MiniLM-L-12-v2` via **FlashRank**
- Runs **100% locally** — no API calls, no cost
- Cross-encoder reads question + chunk together for higher accuracy
- Latency: ~50–150ms per query

### Layer 3: Confidence Filtering

Chunks below the `confidence_threshold` (default: `0.3`) are dropped. If all chunks fail, the system returns **"I don't know"** instead of hallucinating.

---

## Supported File Types

- `.pdf` — PDF documents (text-based and scanned)
- `.docx` — Microsoft Word documents
- `.txt` — Plain text files
- `.md` — Markdown files
- Web URLs — Content fetched and chunked at ingest time

---

## API Endpoints

### Health Check
```
GET /health
→ 200 OK { "status": "healthy" }
```

### Ingest Document
```
POST /ingest
Content-Type: multipart/form-data
Body: file=<binary>
→ 201 { "doc_id": "uuid", "status": "ready", "chunk_count": 42 }
```

### Query Documents
```
POST /query
Content-Type: application/json
Body: { "question": "What are the payment terms?", "doc_ids": ["uuid-1"] }
→ 200 { "answer": "...", "has_relevant_context": true, "sources": [...] }
```

### List Documents
```
GET /documents
→ 200 [{ "doc_id": "...", "filename": "...", "chunk_count": 42 }]
```

### Delete Document
```
DELETE /documents/{doc_id}
→ 200 { "deleted": true, "chunks_removed": 42 }
```

---

## Configuration

All settings are controlled via environment variables in `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
CHROMA_HOST=127.0.0.1
CHROMA_PORT=8001
CONFIDENCE_THRESHOLD=0.3
RETRIEVAL_TOP_K=10
RERANK_TOP_N=3
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Avg. ingest time (10-page PDF) | ~8 seconds |
| Avg. query latency (with rerank) | ~2.1 seconds |
| Avg. query latency (no rerank) | ~0.9 seconds |
| Max tested document size | 500 pages |
| Max tested collection size | 50,000 chunks |

---

## Known Limitations

1. **Scanned PDFs** require OCR pre-processing — not handled automatically.
2. **Tables in PDFs** may lose formatting during text extraction.
3. **Changing embedding models** after ingest requires full re-ingestion.
4. **BM25 index** is rebuilt on every query — cache recommended for >10k chunks.

---

## Changelog

### v0.1.0 (2024-01-01)
- Initial release
- PDF, DOCX, TXT, MD ingestion support
- Three-layer RAG pipeline
- LangGraph multi-step agent

### v0.2.0 (planned)
- Web URL ingestion
- Streaming responses via SSE
- Postgres-backed document registry
