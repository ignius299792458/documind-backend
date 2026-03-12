"""
core/retrieval/vectorstore.py
------------------------------
Manages the ChromaDB vector store — the persistent database that stores
document chunks as high-dimensional embedding vectors.

WHAT THIS MODULE OWNS:
  - Creating / connecting to the ChromaDB collection
  - Embedding model setup (OpenAI)
  - Writing chunks to ChromaDB  (add_chunks_to_vectorstore)
  - Deleting chunks by doc_id   (remove_document)
  - Reading chunk counts        (get_chunk_count)
  - Listing stored documents    (list_documents)

WHAT THIS MODULE DOES NOT OWN:
  - Retrieval logic (that lives in retrievers.py)
  - Chunking (splitters.py)
  - Loading files (loaders.py)

HOW VECTOR SEARCH WORKS IN ONE PARAGRAPH:
  At ingest time every chunk of text is passed to an embedding model
  (OpenAI text-embedding-3-small) which returns a list of 1,536 floats —
  a point in 1,536-dimensional space. Semantically similar text lands
  near the same region of that space. ChromaDB stores those points on disk
  along with the original text and metadata. At query time the user's
  question is embedded the same way, and ChromaDB finds the N stored
  points whose vectors are closest (cosine similarity). Those are the
  "most relevant" chunks.

SINGLETON PATTERN:
  Both the embedding model and the vectorstore are created once via
  @lru_cache and reused across every request. Creating them per-request
  would be slow (OpenAI client init + disk I/O) and waste memory.

CHROMADB PERSISTENCE:
  ChromaDB writes a SQLite database + binary HNSW index files to
  settings.chroma_persist_dir (default: ./chroma_db).
  On restart the existing index is loaded — no re-embedding needed.
"""

import logging
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from documind_backend.config import settings

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding model
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Return a cached OpenAI embeddings client.

    MODELS COMPARISON:
    ┌──────────────────────────┬────────────┬──────────────────┬────────────┐
    │ Model                    │ Dimensions │ Cost/1M tokens   │ Quality    │
    ├──────────────────────────┼────────────┼──────────────────┼────────────┤
    │ text-embedding-3-small   │ 1,536      │ $0.02            │ Good       │
    │ text-embedding-3-large   │ 3,072      │ $0.13            │ Best       │
    │ text-embedding-ada-002   │ 1,536      │ $0.10            │ Legacy     │
    └──────────────────────────┴────────────┴──────────────────┴────────────┘

    Recommendation:
      - text-embedding-3-small  for cost-sensitive / high-volume use cases
      - text-embedding-3-large  when retrieval accuracy is the top priority

    Configured via EMBEDDING_MODEL in .env (default: text-embedding-3-small).

    NOTE: If you change the embedding model AFTER documents are already
    ingested, those chunks have 3-small vectors but new chunks get 3-large
    vectors — they live in incompatible spaces and similarity search breaks.
    Always re-ingest all documents after changing the embedding model.
    """
    logger.info(f"[vectorstore] Initializing embedding model: {settings.openai_embedding_model}")
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
        # Retry on transient OpenAI errors (rate limits, timeouts)
        max_retries=3,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB vectorstore
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    Return a cached Chroma vectorstore connected to the persistent collection.

    CHROMA INTERNALS (simplified):
      - Uses HNSW (Hierarchical Navigable Small World) for approximate
        nearest-neighbor search. Fast and memory-efficient.
      - Metadata filters use DuckDB under the hood — fast SQL-like filtering.
      - The persist_directory holds:
          chroma.sqlite3          ← document metadata + metadata index
          <collection-uuid>/      ← HNSW binary index files

    COLLECTION NAMING:
      All DocuMind documents share one collection (documind_docs).
      Per-document isolation is achieved via metadata filtering:
          {"doc_id": "uuid-123"}
      This is simpler than one-collection-per-document and lets us
      do cross-document search without any extra setup.
    """
    logger.info(
        f"[vectorstore] Connecting to ChromaDB | "
        f"path='{settings.chroma_persist_dir}' | "
        f"collection='{settings.chroma_collection_name}'"
    )
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
        # collection_metadata sets index-level settings.
        # hnsw:space=cosine → use cosine similarity (standard for text).
        # Other options: "l2" (Euclidean), "ip" (inner product).
        collection_metadata={"hnsw:space": "cosine"},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Write operations
# ══════════════════════════════════════════════════════════════════════════════

def add_chunks_to_vectorstore(chunks: list[Document]) -> list[str]:
    """
    Embed a list of text chunks and persist them in ChromaDB.

    Called once per document during the ingestion pipeline, after
    splitters.split_documents() produces the final chunk list.

    WHAT HAPPENS INTERNALLY:
      1. For each chunk, calls OpenAI Embeddings API to get a 1536-dim vector.
         This is batched automatically — one API call per ~2048 chunks.
      2. Stores (vector, page_content, metadata, auto-generated-id) in Chroma.
      3. Chroma persists to disk immediately (no explicit .persist() needed
         in Chroma >= 0.4.x).

    COST ESTIMATE:
      text-embedding-3-small: $0.02 per 1M tokens
      A typical 20-page PDF ≈ 100 chunks ≈ ~15,000 tokens ≈ $0.0003
      A 200-page PDF ≈ 1,000 chunks ≈ ~150,000 tokens ≈ $0.003

    Args:
        chunks: Output of splitters.split_documents(). Each Document must
                have doc_id in its metadata (set by loaders.py).

    Returns:
        List of Chroma-assigned string IDs for the stored chunks.
        Store these if you want to delete specific chunks later.
        (For full document deletion, use remove_document(doc_id) instead.)

    Raises:
        RuntimeError: If the embedding API call fails after retries.
    """
    if not chunks:
        logger.warning("[vectorstore] add_chunks_to_vectorstore called with empty list")
        return []

    doc_id   = chunks[0].metadata.get("doc_id", "unknown")
    filename = chunks[0].metadata.get("filename", "unknown")

    logger.info(
        f"[vectorstore] Embedding {len(chunks)} chunks | "
        f"doc_id={doc_id} | file='{filename}'"
    )

    vectorstore = get_vectorstore()

    try:
        # add_documents() calls the embedding model and writes to Chroma.
        # Returns a list of auto-generated UUIDs (one per chunk).
        ids = vectorstore.add_documents(chunks)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to embed/store chunks for '{filename}': {exc}"
        ) from exc

    logger.info(
        f"[vectorstore] Stored {len(ids)} chunks for doc_id={doc_id}"
    )
    return ids


# ══════════════════════════════════════════════════════════════════════════════
# Delete operations
# ══════════════════════════════════════════════════════════════════════════════

def remove_document(doc_id: str) -> int:
    """
    Delete ALL chunks belonging to a document from ChromaDB.

    Uses ChromaDB's metadata filter to find chunks where
    metadata.doc_id == doc_id, then deletes them by their IDs.

    This is the clean way to "un-ingest" a document — after calling this,
    queries will no longer find any content from that document.

    Args:
        doc_id: The UUID of the document to remove (same UUID used at ingest).

    Returns:
        Number of chunks deleted.

    Raises:
        ValueError: If no chunks are found for the given doc_id.
    """
    vectorstore = get_vectorstore()
    collection  = vectorstore._collection   # access raw ChromaDB collection

    # Fetch all chunk IDs for this document (no content needed, just IDs)
    results = collection.get(
        where={"doc_id": doc_id},
        include=[],   # empty list = return only IDs, no content/vectors
    )

    chunk_ids = results.get("ids", [])

    if not chunk_ids:
        raise ValueError(
            f"No chunks found for doc_id='{doc_id}'. "
            "Document may not exist or was already deleted."
        )

    # Delete all chunks for this document in one call
    collection.delete(ids=chunk_ids)

    logger.info(
        f"[vectorstore] Deleted {len(chunk_ids)} chunks for doc_id={doc_id}"
    )
    return len(chunk_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Read / inspection operations
# ══════════════════════════════════════════════════════════════════════════════

def get_chunk_count(doc_id: str | None = None) -> int:
    """
    Return the number of chunks stored in ChromaDB.

    Args:
        doc_id: If provided, count only chunks for that specific document.
                If None, count ALL chunks across all documents.

    Returns:
        Integer count of matching chunks.

    Example:
        >>> get_chunk_count()               # total in DB, e.g. 4821
        >>> get_chunk_count("uuid-123")     # chunks for one doc, e.g. 63
    """
    vectorstore = get_vectorstore()
    collection  = vectorstore._collection

    if doc_id:
        results = collection.get(
            where={"doc_id": doc_id},
            include=[],   # IDs only — fastest query
        )
        return len(results.get("ids", []))

    # No filter → total count in the collection
    return collection.count()


def list_documents() -> list[dict[str, Any]]:
    """
    Return a summary list of all unique documents currently in ChromaDB.

    Scans all stored chunk metadata and deduplicates by doc_id to produce
    one entry per document. Used by GET /documents to show the document
    library without needing a separate SQL database.

    Returns:
        List of dicts, one per unique document, sorted by filename:
        [
            {
                "doc_id":      "uuid-123",
                "filename":    "report.pdf",
                "file_type":   "pdf",
                "total_pages": 12,
                "chunk_count": 63,
            },
            ...
        ]

    NOTE: For large collections (10,000+ chunks) this full scan is slow.
    In production, maintain a separate documents table in Postgres
    and use this only as a fallback.
    """
    vectorstore = get_vectorstore()
    collection  = vectorstore._collection

    # Fetch all stored metadata (no vectors or text needed)
    results = collection.get(include=["metadatas"])
    metadatas = results.get("metadatas") or []

    if not metadatas:
        return []

    # Deduplicate: group chunks by doc_id, keep first metadata seen per doc
    seen: dict[str, dict[str, Any]] = {}
    chunk_counts: dict[str, int] = {}

    for meta in metadatas:
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue

        chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1

        # First time we see this doc_id → record its metadata
        if doc_id not in seen:
            seen[doc_id] = {
                "doc_id":      doc_id,
                "filename":    meta.get("filename",    "unknown"),
                "file_type":   meta.get("file_type",   "unknown"),
                "total_pages": meta.get("total_pages", None),
            }

    # Merge chunk counts into each document summary
    documents = []
    for doc_id, doc_meta in seen.items():
        documents.append({
            **doc_meta,
            "chunk_count": chunk_counts.get(doc_id, 0),
        })

    # Sort alphabetically by filename for consistent UI ordering
    documents.sort(key=lambda d: d["filename"].lower())

    logger.info(
        f"[vectorstore] list_documents → {len(documents)} unique document(s) | "
        f"{sum(chunk_counts.values())} total chunks"
    )
    return documents


def document_exists(doc_id: str) -> bool:
    """
    Check whether a document with the given doc_id has any chunks in ChromaDB.

    Faster than get_chunk_count() because it stops at the first result.
    Used by the ingest route to detect duplicate uploads.

    Args:
        doc_id: UUID to check.

    Returns:
        True if at least one chunk exists for this doc_id, False otherwise.
    """
    vectorstore = get_vectorstore()
    collection  = vectorstore._collection

    results = collection.get(
        where={"doc_id": doc_id},
        limit=1,       # stop after finding the first match
        include=[],    # IDs only
    )
    return len(results.get("ids", [])) > 0