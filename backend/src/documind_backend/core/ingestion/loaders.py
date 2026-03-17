"""
core/ingestion/loaders.py
--------------------------
Converts uploaded files into LangChain Document objects.

A LangChain Document is a simple container:
    Document(page_content="raw text here", metadata={"key": "value"})

RESPONSIBILITY OF THIS MODULE:
    file on disk  →  List[Document]  (one Document per page/section)

The next step (splitters.py) takes these Documents and breaks them
into smaller chunks. Keeping loading and splitting separate makes
each step independently testable and swappable.

LOADER SELECTION STRATEGY:
    .pdf   → PyPDFLoader       — extracts text per page, preserves page numbers
    .docx  → Docx2txtLoader   — strips formatting, returns plain text
    .txt   → TextLoader        — reads raw text, detects encoding
    .md    → TextLoader        — markdown is plain text, splitter handles structure
    .html  → BSHTMLLoader      — BeautifulSoup strips tags, keeps readable text
    URL    → WebBaseLoader     — fetches + parses web page via HTTP

All loaders are funneled through a single load_document() dispatcher
so the rest of the app never needs to know which loader handles what.
"""

import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from langchain_community.document_loaders import (
    PyPDFLoader,  # pip: pypdf
    Docx2txtLoader,  # pip: python-docx
    TextLoader,  # built into langchain-community
    BSHTMLLoader,  # pip: beautifulsoup4
    WebBaseLoader,  # pip: beautifulsoup4 + httpx
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Extension → Loader class mapping
# To support a new file type: add the extension here + handle it below.
# ──────────────────────────────────────────────────────────────────────────────
EXTENSION_LOADER_MAP: dict[str, type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}


def load_document(
    file_path: str,
    doc_id: str,
    filename: str,
    extra_metadata: Optional[dict] = None,
) -> list[Document]:
    """
    Load a file from disk into a list of LangChain Documents.

    Each Document in the returned list represents one logical section
    of the source file:
      - PDF  → one Document per PAGE  (PyPDFLoader splits by page automatically)
      - DOCX → one Document for the ENTIRE file (no natural page boundaries)
      - TXT  → one Document for the ENTIRE file
      - HTML → one Document with all visible text extracted

    Every Document gets enriched metadata so downstream steps (retrieval,
    citations) know exactly where each chunk came from.

    Args:
        file_path:      Absolute or relative path to the file on disk.
        doc_id:         UUID assigned to this document at upload time.
                        Stored in every chunk's metadata for filtering:
                        "give me only chunks from THIS document".
        filename:       Original filename shown in the UI citation panel.
        extra_metadata: Any additional key-value pairs to attach to every
                        Document. Useful for tagging (e.g. {"category": "legal"}).

    Returns:
        Non-empty list of Document objects ready for the splitter.

    Raises:
        FileNotFoundError: File doesn't exist at file_path.
        ValueError:        File extension is not in EXTENSION_LOADER_MAP.
        RuntimeError:      Loader failed to parse the file content.

    Example:
        >>> docs = load_document("/tmp/report.pdf", "uuid-123", "report.pdf")
        >>> print(len(docs))          # one per page, e.g. 12
        >>> print(docs[0].metadata)   # {"doc_id": "uuid-123", "page": 1, ...}
    """
    path = Path(file_path)

    # ── Guard: file must exist ───────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(f"Cannot load document — file not found: '{file_path}'")

    # ── Guard: extension must be supported ───────────────────────────────────
    extension = path.suffix.lower()
    if extension not in EXTENSION_LOADER_MAP:
        supported = ", ".join(sorted(EXTENSION_LOADER_MAP.keys()))
        raise ValueError(
            f"Unsupported file extension '{extension}'. " f"Supported formats: {supported}"
        )

    logger.info(f"[loaders] Loading '{filename}' | " f"doc_id={doc_id} | format={extension}")

    # ── Load ─────────────────────────────────────────────────────────────────
    # Each loader's .load() returns List[Document].
    # PyPDFLoader: one Document per page (metadata includes 'page' key).
    # All others: typically one Document for the whole file.
    try:
        loader_class = EXTENSION_LOADER_MAP[extension]

        # TextLoader needs explicit encoding — UTF-8 covers most files,
        # autodetect=True silently falls back for Windows-encoded files.
        if extension in (".txt", ".md"):
            loader = loader_class(file_path, encoding="utf-8", autodetect_encoding=True)
        else:
            loader = loader_class(file_path)

        documents = loader.load()

    except Exception as exc:
        raise RuntimeError(f"Loader failed to parse '{filename}': {exc}") from exc

    if not documents:
        raise RuntimeError(
            f"Loader returned zero documents for '{filename}'. "
            "The file may be empty or its text is not extractable "
            "(e.g. a scanned image PDF with no OCR layer)."
        )

    # ── Enrich metadata ──────────────────────────────────────────────────────
    # Metadata is stored alongside every chunk in ChromaDB.
    # It enables two critical features:
    #   1. Per-document filtering:  {"doc_id": "uuid-123"}
    #   2. UI source citations:     "filename.pdf, page 4"
    base_metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "file_type": extension.lstrip("."),  # "pdf" not ".pdf"
        "total_pages": len(documents),
        **(extra_metadata or {}),
    }

    for i, doc in enumerate(documents):
        # Merge base metadata INTO each document's existing metadata.
        # Use dict unpacking so we don't overwrite keys the loader already set
        # (e.g. PyPDFLoader sets its own 'page' key — we want to keep that).
        doc.metadata = {**base_metadata, **doc.metadata}

        # Guarantee a 'page' key exists even for loaders that don't set one.
        # 1-indexed (page 1, page 2, ...) for human-readable citations.
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1

        # Normalize PyPDFLoader's 0-indexed page to 1-indexed.
        # PyPDFLoader sets page=0 for the first page, which is confusing in UI.
        elif extension == ".pdf":
            doc.metadata["page"] = doc.metadata["page"] + 1

    logger.info(f"[loaders] Loaded {len(documents)} page(s) from '{filename}'")
    return documents


async def load_from_url(
    url: str,
    doc_id: str,
    extra_metadata: Optional[dict] = None,
) -> list[Document]:
    """
    Fetch a public web page and load its text content as Documents.

    Uses WebBaseLoader which:
    1. Makes an HTTP GET request to the URL
    2. Parses the HTML response with BeautifulSoup4
    3. Strips all tags, keeping only visible text
    4. Returns the cleaned text as one Document

    This is useful for ingesting documentation sites, blog posts,
    or any publicly accessible web content.

    Args:
        url:            The full URL to fetch (must be publicly accessible,
                        no authentication supported).
        doc_id:         UUID to tag all resulting Documents with.
        extra_metadata: Additional metadata to attach.

    Returns:
        List of Documents (usually one per page/section discovered).

    Raises:
        RuntimeError: If the URL cannot be fetched or parsed.

    Example:
        >>> docs = await load_from_url("https://docs.python.org/3/", "uuid-456")
        >>> print(docs[0].metadata["source_url"])
    """
    logger.info(f"[loaders] Fetching URL: {url} | doc_id={doc_id}")

    try:
        # WebBaseLoader is synchronous internally — wrap in asyncio if needed
        # for high-throughput scenarios. For now, direct call is fine.
        loader = WebBaseLoader(
            web_paths=[url],
            # bs_kwargs are passed directly to BeautifulSoup.
            # Passing None uses the default parser (html.parser),
            # which handles malformed HTML gracefully.
            # bs_kwargs={"features": "html.parser"},
        )
        documents = loader.load()

    except Exception as exc:
        raise RuntimeError(f"Failed to fetch URL '{url}': {exc}") from exc

    if not documents:
        raise RuntimeError(f"No content extracted from URL: '{url}'")

    # Enrich metadata
    for i, doc in enumerate(documents):
        doc.metadata.update(
            {
                "doc_id": doc_id,
                "filename": url,  # use URL as display name in citations
                "file_type": "url",
                "source_url": url,
                "page": i + 1,
                "total_pages": len(documents),
                **(extra_metadata or {}),
            }
        )

    logger.info(f"[loaders] Loaded {len(documents)} section(s) from URL: {url}")
    return documents


def get_supported_extensions() -> list[str]:
    """
    Return a sorted list of supported file extensions.
    Used by the upload validation layer to reject unsupported file types.

    Example:
        >>> get_supported_extensions()
        ['.docx', '.htm', '.html', '.md', '.pdf', '.txt']
    """
    return sorted(EXTENSION_LOADER_MAP.keys())


if __name__ == "__main__":
    # Get supported extensions
    print("Supported extensions:", get_supported_extensions())

    # Quick test: load a sample PDF and print the first Document's metadata
    sample_path = (
        "/Users/maheshbogati/Documents/langchain/documind/documind_backend/dependencies.md"
    )
    sample_docs = load_document(sample_path, str(uuid4()), "dependencies.md")
    print(sample_docs[0].metadata)
    file_type = sample_docs[0].metadata.get("file_type")

    # splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(sample_docs)
    import documind_backend.core.ingestion.splitters as splitters

    chunks = splitters.split_documents(sample_docs, str(file_type))
    print(
        f"Split into {len(chunks)} chunks. \n First chunk metadata: {chunks[0].metadata} \n First chunk content: {chunks[0].page_content[:200]}..."
    )

    # Quick test: fetch a web page and print the first Document's metadata
    import asyncio

    sample_url = "https://docs.langchain.com"
    sample_web_docs = asyncio.run(load_from_url(sample_url, str(uuid4())))
    print(sample_web_docs[0].metadata)
