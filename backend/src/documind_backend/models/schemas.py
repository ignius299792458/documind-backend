"""
models/schemas.py
-----------------
All Pydantic models used for:
  - FastAPI request/response validation
  - Internal data contracts between layers
  - LangChain output parsing

Keeping all schemas in one file makes it easy to see the full
data contract of the application at a glance.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

# ================================================================== #
# Enums
# ================================================================== #


class DocumentStatus(str, Enum):
    """Lifecycle states a document moves through after upload."""

    PENDING = "pending"  # uploaded, not yet processed
    INGESTING = "ingesting"  # currently being chunked + embedded
    READY = "ready"  # in vector store, queryable
    FAILED = "failed"  # processing failed — see error field


class SupportedFileType(str, Enum):
    """
    File types DocuMind can ingest.
    Extending support: add the mime type here and handle it in loaders.py.
    """

    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    TXT = "text/plain"
    MD = "text/markdown"
    HTML = "text/html"


# ================================================================== #
# Document models
# ================================================================== #


class DocumentMeta(BaseModel):
    """
    Metadata stored alongside every document.
    This is persisted in ChromaDB as metadata on each chunk,
    so we can filter by doc_id, file_type, or upload_date at query time.
    """

    doc_id: str = Field(..., description="UUID assigned at upload time")
    filename: str
    file_type: str
    file_size_bytes: int
    page_count: Optional[int] = None  # None for non-paginated formats like .txt
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    status: DocumentStatus = DocumentStatus.PENDING
    error: Optional[str] = None  # populated if status == FAILED
    chunk_count: Optional[int] = None  # populated after ingestion completes


class DocumentListResponse(BaseModel):
    """Response body for GET /documents."""

    documents: list[DocumentMeta]
    total: int


class DeleteResponse(BaseModel):
    """Response body for DELETE /documents/{doc_id}."""

    doc_id: str
    chunks_deleted: int
    message: str

class DeleteAllResponse(BaseModel):
    """Delete all responses"""
    total_deleted_docs: int
    message: str

# ================================================================== #
# Ingestion models
# ================================================================== #


class IngestResponse(BaseModel):
    """
    Returned immediately after POST /ingest.
    Processing happens asynchronously — poll GET /documents/{doc_id}
    to check when status becomes READY.
    """

    doc_id: str
    filename: str
    status: DocumentStatus
    message: str


# ================================================================== #
# Query / Chat models
# ================================================================== #


class QueryRequest(BaseModel):
    """
    Request body for POST /query.

    doc_ids: if provided, restrict retrieval to those specific documents.
             if empty/None, search across all documents in the collection.
    """

    question: str = Field(..., min_length=1, max_length=2000)
    doc_ids: Optional[list[str]] = None
    # Number of source citations to include in the response
    top_k: int = Field(default=3, ge=1, le=10)


class SourceChunk(BaseModel):
    """
    A retrieved document chunk returned alongside the answer.
    Shown in the UI as a citation the user can expand to verify.
    """

    doc_id: str
    filename: str
    page: Optional[int] = None  # page number within the source document
    chunk_index: int  # which chunk within that document
    content: str  # the actual text of this chunk
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """
    Full response for non-streaming POST /query calls.
    Streaming endpoint (SSE) sends tokens directly — see query.py.
    """
    question: str 
    answer: str
    sources: list[SourceChunk]
    # True if the retriever found relevant context.
    # False means the answer is "I don't know" — no hallucination.
    has_relevant_context: bool
    # Total prompt + completion tokens used (for cost tracking)
    tokens_used: Optional[int] = None


class URLIngestRequest(BaseModel):
    """Request body for POST /ingest/url."""

    url: str
    # Optional display name — if not provided, the URL is used as filename
    display_name: str | None = None


# ================================================================== #
# Agent models
# ================================================================== #


class AgentQueryRequest(BaseModel):
    """
    Request body for POST /agent/query — uses the LangGraph agent
    instead of the simple RAG chain. Use this for complex,
    multi-step questions that require multiple retrieval passes.
    """

    question: str = Field(..., min_length=1, max_length=2000)
    doc_ids: Optional[list[str]] = None
    # session_id enables conversation memory across multiple turns.
    # Pass the same session_id in follow-up questions to maintain context.
    session_id: Optional[str] = Field(
        default=None, description="Pass the same ID across turns to maintain conversation memory"
    )


class AgentQueryResponse(BaseModel):
    """Response from the LangGraph agent endpoint."""

    answer: str
    session_id: str  # echo back so the client can use it next turn
    steps_taken: int  # how many retrieval steps the agent performed
    sources: list[SourceChunk]
    has_relevant_context: bool


class BatchQueryRequest(BaseModel):
    """Request body for POST /query/batch."""

    questions: list[str]
    doc_ids: list[str] | None = None

    # Max questions per batch — prevent abuse
    # Validated by pydantic at field level
    class Config:
        json_schema_extra = {
            "example": {
                "questions": [
                    "What are the payment terms?",
                    "When does the contract expire?",
                    "Who are the signatories?",
                ],
                "doc_ids": None,
            }
        }


class BatchQueryResponse(BaseModel):
    """Response body for POST /query/batch."""

    results: list[QueryResponse]
    total: int


# ================================================================== #
# Health check
# ================================================================== #


class HealthResponse(BaseModel):
    """Response for GET /health — used by load balancers and uptime monitors."""

    status: str = "ok"
    version: str = "0.1.0"
    environment: str
    vector_store_chunk_count: int  # total chunks currently in ChromaDB


class ConfigurationModelParameterResponse(BaseModel):
    """Response body for GET /admin/config/model-params"""
    is_plugged: bool
    
class ConfigurationModelParameterBody(BaseModel):
    """Response body for GET /admin/config/model-params"""
    ollama_creativity_temperature: float = Field(..., ge=0.0, le=1.0)
    retrieval_top_k: int =  Field(..., ge=5, le=20)
    rerank_top_n: int =  Field(..., ge=1, le=10)
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)