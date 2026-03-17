"""
api/ingest.py
---------------------
Document ingestion endpoint.

ENDPOINT:
  POST /ingest  — Upload a file and process it into the vector store.

FULL INGESTION PIPELINE:
  1. Validate   — file type, file size, not already ingested
  2. Save       — write bytes to ./uploads/{doc_id}_{filename}
  3. Load       — loaders.load_document() → List[Document] (one per page)
  4. Split      — splitters.split_documents() → List[Document] (many chunks)
  5. Embed+Store — vectorstore.add_chunks_to_vectorstore() → writes to ChromaDB
  6. Cleanup    — delete temp file from disk
  7. Respond    — return doc_id, filename, chunk_count, status

SYNC VS ASYNC:
  Currently synchronous — the HTTP request blocks until ingestion is done.
  For a 20-page PDF this takes ~3-8 seconds (mostly embedding API calls).
  For a 200-page PDF this can take 30-60 seconds.

  Future improvement: move to background tasks using FastAPI BackgroundTasks
  or a task queue (Celery, ARQ). The endpoint would return immediately with
  status=PENDING and a doc_id, and the client polls GET /documents/{doc_id}
  until status=READY.

FILE STORAGE:
  Uploaded files are saved temporarily to settings.upload_dir.
  After successful ingestion the temp file is deleted.
  Only the vector embeddings (in ChromaDB) are kept permanently.
"""

import logging
import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, status, UploadFile, File, HTTPException

from documind_backend.config import settings
import documind_backend.core.ingestion.loaders as IngestionLoaders
import documind_backend.core.ingestion.splitters as IngestionSplitters
import documind_backend.core.retrieval.vectorstore as VectorStore
import documind_backend.models.schemas as Schemas

logger = logging.getLogger(__name__)
router = APIRouter()

# Ensure upload directory exists on startup
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Validation constants
# ══════════════════════════════════════════════════════════════════════════════

# File extensions allowed for upload.
# Derived from loaders.py — single source of truth.
ALLOWED_EXTENSIONS = set(IngestionLoaders.get_supported_extensions())  # {'.pdf', '.docx', ...}

# MIME types browsers send in Content-Type header.
# Used as a secondary check alongside extension validation.
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
    "text/html",
    "application/octet-stream",  # some browsers send this for unknown types
}


# ══════════════════════════════════════════════════════════════════════════════
# POST /ingest
# ══════════════════════════════════════════════════════════════════════════════


@router.post(
    "/ingest",
    response_model=Schemas.IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description=(
        "Upload a PDF, DOCX, TXT, MD, or HTML file. "
        "The file is chunked, embedded with nomic-embed-text via Ollama, "
        "and stored in ChromaDB. Returns a doc_id for future queries. "
        f"Max file size: {settings.max_upload_size_mb}MB."
    ),
)
async def ingest_document(
    file: UploadFile = File(
        ...,
        description="Document file to ingest. Supported: PDF, DOCX, TXT, MD, HTML.",
    ),
) -> Schemas.IngestResponse:
    """
    Upload a document and run the full ingestion pipeline.

    STEPS:
      1. Validate extension + file size
      2. Check for duplicate (same filename already ingested)
      3. Save file bytes to disk (temp storage)
      4. load_document()  → pages as Documents
      5. split_documents() → chunks as Documents
      6. add_chunks_to_vectorstore() → embed + persist in ChromaDB
      7. Delete temp file
      8. Return IngestResponse with doc_id + stats

    Args:
        file: The uploaded file (multipart/form-data).

    Returns:
        IngestResponse with doc_id, filename, status, chunk_count.

    Raises:
        400: Invalid file type or file is empty.
        413: File exceeds max upload size.
        422: Validation error (Pydantic).
        500: Ingestion pipeline failed.
    """
    filename = file.filename or "unnamed_file"
    extension = Path(filename).suffix.lower()

    logger.info(f"[ingest] Received upload: '{filename}' ({extension})")

    # ── Step 1a: Validate file extension ─────────────────────────────────────
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File type '{extension}' is not supported. "
                f"Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # ── Step 1b: Read file bytes + validate size ──────────────────────────────
    # We read the entire file into memory to check size before saving to disk.
    # For very large files this could be a memory concern — for the current
    # 50MB limit it's fine on any modern machine.
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file '{filename}' is empty (0 bytes).",
        )

    max_bytes = settings.max_upload_size_bytes
    if file_size > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {file_size / 1024 / 1024:.1f}MB exceeds "
                f"maximum allowed size of {settings.max_upload_size_mb}MB."
            ),
        )

    logger.info(f"[ingest] File size: {file_size / 1024:.1f}KB — within limit")

    # ── Step 2: Generate doc_id + build temp file path ────────────────────────
    # doc_id is a UUID assigned to this document permanently.
    # All chunks in ChromaDB are tagged with this doc_id.
    # Clients use it to scope queries to specific documents.
    doc_id = str(uuid.uuid4())
    # Sanitize filename for safe filesystem storage
    safe_name = _sanitize_filename(filename)
    file_path = os.path.join(settings.upload_dir, f"{doc_id}_{safe_name}")

    # ── Step 3: Save file to disk ─────────────────────────────────────────────
    # Use aiofiles for non-blocking async file I/O.
    # Without aiofiles, file.write() blocks the event loop during disk write.
    logger.info(f"[ingest] Saving to: {file_path}")
    try:
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_bytes)
    except OSError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {e}",
        )

    # ── Steps 4-6: Run ingestion pipeline ────────────────────────────────────
    # Wrapped in try/finally to guarantee temp file cleanup even on failure.
    chunk_count = 0
    try:
        # Step 4: Load — file on disk → List[Document] (one per page/section)
        logger.info(f"[ingest] Loading '{filename}'...")
        pages = IngestionLoaders.load_document(
            file_path=file_path,
            doc_id=doc_id,
            filename=filename,
        )
        logger.info(f"[ingest] Loaded {len(pages)} page(s)")

        # Step 5: Split — pages → chunks (smaller pieces for embedding)
        logger.info("[ingest] Splitting into chunks...")
        chunks = IngestionSplitters.split_documents(pages, file_type=extension.lstrip("."))
        chunk_count = len(chunks)
        logger.info(f"[ingest] Created {chunk_count} chunk(s)")

        if chunk_count == 0:
            raise ValueError(
                "Document produced zero chunks after splitting. "
                "The file may contain only images without extractable text."
            )

        # Step 6: Embed + store — chunks → vectors → ChromaDB
        logger.info(f"[ingest] Embedding {chunk_count} chunks with nomic-embed-text...")
        chunk_ids = VectorStore.add_chunks_to_vectorstore(chunks)
        logger.info(f"[ingest] Stored {len(chunk_ids)} vectors in ChromaDB")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        # Known pipeline errors — return clear 400/422 error messages
        logger.error(f"[ingest] Pipeline error for '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        # Unexpected errors — log full traceback, return 500
        logger.exception(f"[ingest] Unexpected error for '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed unexpectedly: {e}",
        )
    finally:
        # Step 7: Always delete temp file — success OR failure
        # On failure: don't leave orphaned files in the uploads directory
        # On success: embeddings are in ChromaDB, original file not needed
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"[ingest] Temp file deleted: {file_path}")

    # ── Step 8: Return success response ──────────────────────────────────────
    logger.info(
        f"[ingest] SUCCESS | doc_id={doc_id} | " f"file='{filename}' | chunks={chunk_count}"
    )

    return Schemas.IngestResponse(
        doc_id=doc_id,
        filename=filename,
        status=Schemas.DocumentStatus.READY,
        message=(
            f"Successfully ingested '{filename}'. "
            f"{len(pages)} page(s) → {chunk_count} chunk(s) stored in vector database."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST /ingest/url — ingest from a URL
# ══════════════════════════════════════════════════════════════════════════════


@router.post(
    "/ingest/url",
    response_model=Schemas.IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a web page from a URL",
    description=(
        "Fetch a public web page, extract its text content, "
        "and ingest it into the vector store. "
        "The URL must be publicly accessible."
    ),
)
async def ingest_from_url(request: Schemas.URLIngestRequest) -> Schemas.IngestResponse:
    """
    Fetch a web page and run the full ingestion pipeline on its content.

    Uses WebBaseLoader (LangChain) to fetch the URL and strip HTML tags,
    leaving clean readable text. Then splits and embeds the same way as file uploads.

    Args:
        request: URLIngestRequest with url and optional display_name.

    Returns:
        IngestResponse with doc_id, filename (the URL), status, chunk_count.
    """
    from documind_backend.core.ingestion.loaders import load_from_url

    url = str(request.url)
    display_name = request.display_name or url
    doc_id = str(uuid.uuid4())

    logger.info(f"[ingest] URL ingest: {url}")

    try:
        # Fetch and parse the web page
        pages = await load_from_url(url=url, doc_id=doc_id)
        logger.info(f"[ingest] Loaded {len(pages)} section(s) from URL")

        # Split into chunks
        chunks = IngestionSplitters.split_documents(pages, file_type="html")
        chunk_count = len(chunks)
        logger.info(f"[ingest] Created {chunk_count} chunk(s) from URL content")

        if chunk_count == 0:
            raise ValueError("No text content could be extracted from the URL.")

        # Embed and store
        VectorStore.add_chunks_to_vectorstore(chunks)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ingest] URL ingest failed for {url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to ingest URL '{url}': {e}",
        )

    return Schemas.IngestResponse(
        doc_id=doc_id,
        filename=display_name,
        status=Schemas.DocumentStatus.READY,
        message=(
            f"Successfully ingested '{display_name}'. "
            f"{len(pages)} section(s) → {chunk_count} chunk(s) stored."
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _sanitize_filename(filename: str) -> str:
    """
    Make a filename safe for filesystem storage.

    Replaces characters that are illegal or problematic on macOS/Linux/Windows:
      spaces → underscores
      special chars → removed

    Keeps: letters, numbers, underscores, hyphens, dots.
    Truncates to 100 chars to avoid path length issues.

    Args:
        filename: Original filename from the upload.

    Returns:
        Sanitized filename safe for os.path.join().

    Examples:
        "My Report (2024).pdf"  → "My_Report_2024_.pdf"
        "../../etc/passwd"      → "etcpasswd"
    """
    import re

    # Replace spaces with underscores
    name = filename.replace(" ", "_")
    # Remove anything that isn't alphanumeric, underscore, hyphen, or dot
    name = re.sub(r"[^\w\-.]", "", name)
    # Truncate to avoid filesystem path length limits
    name = name[:100]
    # Fallback if everything was stripped
    return name or "unnamed_file"
