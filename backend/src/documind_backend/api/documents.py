"""
api/documents.py
------------------------
Document management endpoints.

ENDPOINTS:
  GET  /documents          → list all ingested documents
  GET  /documents/{doc_id} → get one document's metadata + chunk count
  DELETE /documents/{doc_id} → remove document from vector store
  GET  /health             → health check for load balancers / Docker

These endpoints talk to the vectorstore layer directly —
no LLM calls happen here. Pure metadata and CRUD operations.
"""

import logging
from fastapi import APIRouter, HTTPException, status

from documind_backend.config import settings
from documind_backend.models.schemas import (
    DocumentListResponse,
    DocumentMeta,
    DocumentStatus,
    HealthResponse,
    DeleteResponse,
    DeleteAllResponse,
)
import documind_backend.core.retrieval.vectorstore as VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# GET /documents — list all documents
# ══════════════════════════════════════════════════════════════════════════════
@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
    description=(
        "Returns metadata for every document currently in the vector store. "
        "Scans ChromaDB chunk metadata and deduplicates by doc_id."
    ),
)
async def list_all_documents() -> DocumentListResponse:
    """
    Return a list of all documents currently stored in ChromaDB.

    Builds the list by scanning chunk metadata in ChromaDB and
    deduplicating by doc_id. No separate SQL table needed.

    Each entry includes:
      - doc_id, filename, file_type
      - total_pages (if available from loader)
      - chunk_count (actual chunks stored in ChromaDB)
      - status is always READY (only fully ingested docs are in ChromaDB)
    """
    logger.info("[documents] GET /documents")
    raw_docs = VectorStore.list_documents()

    # Convert raw dicts to DocumentMeta Pydantic models for respone validation
    documents = [
        DocumentMeta(
            doc_id=d["doc_id"],
            filename=d["filename"],
            file_type=d["file_type"],
            file_size_bytes=0,
            page_count=d.get("total_pages"),
            status=DocumentStatus.READY,
            chunk_count=d.get("chunk_count"),
        )
        for d in raw_docs
    ]

    logger.info(f"[documents] Found {len(documents)} document(s)")

    return DocumentListResponse(documents=documents, total=len(documents))


# ══════════════════════════════════════════════════════════════════════════════
# GET /documents/{doc_id} — get one document
# ══════════════════════════════════════════════════════════════════════════════


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentMeta,
    summary="Get a single document's metadata",
    description="Returns metadata and chunk count for one document by its UUID.",
)
async def get_document(doc_id: str) -> DocumentMeta:
    """
    Return metadata for a single document identified by its UUID.

    Looks up the document in ChromaDB by filtering on doc_id metadata.
    Returns 404 if no chunks exist for the given doc_id.

    Args:
        doc_id: The UUID assigned at ingest time.
    """
    logger.info(f"[documents] GET /documents/{doc_id}")

    # Check existence first — avoids scanning full collection unnecessarily
    if not VectorStore.document_exists(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found. It may not have been ingested yet.",
        )

    # Get exact chunk count for this document
    chunk_count = VectorStore.get_chunk_count(doc_id=doc_id)

    # Fetch metadata from the first chunk of this document
    collection = VectorStore.get_vectorstore()._collection
    results = collection.get(
        where={"doc_id": doc_id},
        limit=1,
        include=["metadatas"],
    )

    meta = results["metadatas"][0] if results["metadatas"] else {}

    return DocumentMeta(
        doc_id=doc_id,
        filename=str(meta.get("filename", "unknown")),
        file_type=str(meta.get("file_type", "unknown")),
        file_size_bytes=0,
        page_count=int(str(meta.get("total_pages"))),
        status=DocumentStatus.READY,
        chunk_count=chunk_count,
    )

# ══════════════════════════════════════════════════════════════════════════════
# GET /delete-all — delete all documents
# ══════════════════════════════════════════════════════════════════════════════

@router.delete(
    "/documents/delete-all",
    response_model=DeleteAllResponse,
    summary="Delete all documents from the vector store",
    description=(
        "Permanently removes all documents and chunks from ChromaDB. "
        "The vector store will be empty after this operation. "
        "This action cannot be undone — you must re-ingest files to restore them."
    ),
)
async def delete_all_documents() -> DeleteAllResponse:
    """
    Delete all documents and chunks from ChromaDB.

    Uses ChromaDB's API to delete all entries in the collection.

    After deletion:
      - GET /documents will return an empty list
      - Queries will not return any results until new documents are ingested

    Raises:
        500: If deletion fails.
    """
    logger.info("[DELETE] all documents")
    
    try:
        all_deleted_chunks_count = VectorStore.delete_all_documents()
    except Exception as e:
        logger.error(f"[documents] Failed to delete all: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete all documents: {e}",
        )
    
    return DeleteAllResponse(
        total_deleted_docs=all_deleted_chunks_count,
        message="Successfully deleted all documents",
    )
    
# ══════════════════════════════════════════════════════════════════════════════
# DELETE /documents/{doc_id} — remove document
# ══════════════════════════════════════════════════════════════════════════════


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteResponse,
    summary="Delete a document from the vector store",
    description=(
        "Permanently removes all chunks of a document from ChromaDB. "
        "The document will no longer appear in queries. "
        "This action cannot be undone — you must re-ingest the file to restore it."
    ),
)
async def delete_document(doc_id: str) -> DeleteResponse:
    """
    Delete all chunks belonging to a document from ChromaDB.

    Uses ChromaDB's metadata filter to find all chunks where
    metadata.doc_id == doc_id, then deletes them by their chunk IDs.

    After deletion:
      - The document no longer appears in GET /documents
      - Queries will not return content from this document
      - The original file (if stored locally) is NOT deleted —
        only the vector store entries are removed

    Args:
        doc_id: UUID of the document to delete.

    Raises:
        404: If no chunks exist for this doc_id.
    """
    logger.info(f"[documents] DELETE /documents/{doc_id}")

    if not VectorStore.document_exists(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    try:
        chunks_deleted = VectorStore.remove_document(doc_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"[documents] Failed to delete doc_id={doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {e}",
        )

    logger.info(f"[documents] Deleted {chunks_deleted} chunks for doc_id={doc_id}")

    return DeleteResponse(
        doc_id=doc_id,
        chunks_deleted=chunks_deleted,
        message=f"Successfully deleted {chunks_deleted} chunks for document '{doc_id}'.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /health — health check
# ══════════════════════════════════════════════════════════════════════════════


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns service health status. "
        "Used by Docker HEALTHCHECK, load balancers, and uptime monitors."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Checks:
      1. FastAPI is responding (trivially true if this runs)
      2. ChromaDB is accessible (calls get_chunk_count)
      3. Returns total chunk count for observability

    Returns 200 if healthy, 503 if ChromaDB is unreachable.
    """
    try:
        total_chunks = VectorStore.get_chunk_count()
    except Exception as e:
        logger.error(f"[health] ChromaDB unreachable: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ChromaDB is unreachable: {e}",
        )

    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=settings.app_env,
        vector_store_chunk_count=total_chunks,
    )
    
