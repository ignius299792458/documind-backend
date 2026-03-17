"""
core/ingestion/splitters.py
----------------------------
Breaks large Documents into smaller chunks ready for embedding with Ollama.

RESPONSIBILITY OF THIS MODULE:
    List[Document] (pages)  →  List[Document] (chunks)

WHY CHUNKING IS NECESSARY:
    1. Embedding model token limits
       Ollama's embedding models accept a maximum token limit per call.
       A 20-page PDF may exceed this limit — it must be split.

    2. Retrieval precision
       Embedding an entire document as one vector reduces focus.
       Smaller chunks produce more precise embeddings → better similarity scores.

    3. Context window management
       LLM prompts cannot hold entire long documents.
       After retrieval, only the top 3-5 relevant chunks are injected.

CHUNK SIZE TRADEOFFS:
    Smaller chunks (200-500 chars):
        + Very focused embeddings
        - Lose surrounding context

    Larger chunks (1500-2000 chars):
        + Richer context per chunk
        - Mixed-topic embeddings

    Sweet spot: 800-1200 chars with 150-250 overlap.
    DocuMind defaults: chunk_size=1000, chunk_overlap=200 (configurable in config.py).

OVERLAP EXPLAINED:
    chunk_overlap=200 repeats the last 200 characters of chunk N
    as the first 200 characters of chunk N+1, preserving sentences across boundaries.

STRATEGY BY FILE TYPE:
    .pdf, .docx, .txt, .html → RecursiveCharacterTextSplitter
        Splits paragraphs → sentences → words → characters, preserving semantic coherence.

    .md → MarkdownHeaderTextSplitter (stage 1) + RecursiveCharacterTextSplitter (stage 2)
        Stage 1 splits on headers, injecting hierarchy into metadata.
        Stage 2 handles sections still too long after header splitting.
"""

import logging
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from documind_backend.config import settings

logger = logging.getLogger(__name__)


def split_documents(
    documents: list[Document],
    file_type: str = "txt",
) -> list[Document]:
    """
    Split a list of loaded Documents into smaller chunks.

    This is the main entry point called by the ingestion pipeline.
    It dispatches to the correct splitting strategy based on file_type,
    then post-processes all chunks with consistent metadata.

    Args:
        documents:  Output from loaders.load_document() — one Document per
                    page/section. Each Document can be arbitrarily large.
        file_type:  File extension WITHOUT the dot ('pdf', 'docx', 'md', etc.)
                    Determines which splitting strategy to use.

    Returns:
        A flat list of chunk Documents. All chunks from all input pages
        are combined into one list. Order is preserved:
        page 1 chunks → page 2 chunks → page 3 chunks → ...

        Each chunk has:
            - page_content: the chunk text
            - metadata:     inherited from parent + chunk_index + start_index

    Example:
        >>> # 12-page PDF → 12 Documents → ~60 chunks
        >>> pages = load_document("report.pdf", doc_id, "report.pdf")
        >>> chunks = split_documents(pages, file_type="pdf")
        >>> print(len(chunks))                    # e.g. 63
        >>> print(chunks[0].metadata["page"])     # 1
        >>> print(chunks[0].metadata["chunk_index"])  # 0
    """
    if not documents:
        logger.warning("[splitters] split_documents() called with empty list — returning []")
        return []

    # Normalize: strip leading dot, lowercase ("PDF" → "pdf", ".pdf" → "pdf")
    file_type = file_type.lower().lstrip(".")

    logger.info(
        f"[splitters] Splitting {len(documents)} page(s) | "
        f"type={file_type} | "
        f"chunk_size={settings.chunk_size} | "
        f"overlap={settings.chunk_overlap}"
    )

    # ── Dispatch to the appropriate splitting strategy ───────────────────────
    if file_type == "md":
        # Markdown gets two-stage splitting (headers first, then characters)
        chunks = _split_markdown(documents)
    else:
        # Everything else: PDF, DOCX, TXT, HTML, URL content
        chunks = _split_recursive(documents)

    # ── Post-process: add chunk_index to every chunk ─────────────────────────
    # chunk_index is a sequential integer across ALL chunks of this document.
    # It's used in the UI citation panel: "chunk 7 of 63"
    # and as a stable identifier for deduplication in the retrieval layer.
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        f"[splitters] Split complete: " f"{len(documents)} page(s) → {len(chunks)} chunk(s)"
    )

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Private splitting strategies
# ──────────────────────────────────────────────────────────────────────────────


def _split_recursive(documents: list[Document]) -> list[Document]:
    """
    RecursiveCharacterTextSplitter — the default strategy for most formats.

    HOW IT WORKS:
    The splitter tries each separator in order until chunks are small enough:
        1. "\\n\\n"  → paragraph breaks (preferred — keeps paragraphs whole)
        2. "\\n"     → line breaks
        3. ". "      → sentence ends
        4. " "       → word boundaries
        5. ""        → individual characters (last resort)

    This means it always tries to split at the most natural boundary first.
    A 3000-char paragraph will be split at its internal sentences rather
    than arbitrarily in the middle of a word.

    CONFIG (from settings):
        chunk_size=1000    → target max characters per chunk
        chunk_overlap=200  → chars repeated between adjacent chunks
        add_start_index=True → adds 'start_index' to metadata (char offset
                               from document start — useful for debugging)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Preferred split points in order of preference
        separators=[
            "\n\n",  # paragraph break — most natural boundary
            "\n",  # line break
            ". ",  # sentence end (with space to avoid splitting decimals)
            "? ",  # question end
            "! ",  # exclamation end
            "; ",  # semicolon
            ", ",  # comma (last resort before word splitting)
            " ",  # word boundary
            "",  # character (absolute last resort)
        ],
        # Keep the separator attached to the END of the chunk it belongs to,
        # not the START of the next chunk. Reads more naturally.
        keep_separator=True,
        # Adds 'start_index' to metadata — the character offset of this chunk
        # from the start of the page. Useful for debugging and highlighting.
        add_start_index=True,
    )

    return splitter.split_documents(documents)


def _split_markdown(documents: list[Document]) -> list[Document]:
    """
    Two-stage Markdown splitting strategy.

    STAGE 1 — MarkdownHeaderTextSplitter:
        Splits the document on Markdown headers:
            # Main Title          → Header 1
            ## Section            → Header 2
            ### Subsection        → Header 3

        The header text is stored in the chunk's metadata:
            metadata = {
                "Header 1": "Installation Guide",
                "Header 2": "Prerequisites",
            }

        WHY THIS MATTERS FOR RAG:
        When a chunk is retrieved, its metadata tells the LLM exactly which
        section it came from, even if the header text isn't in the chunk.
        This dramatically improves citation quality.

    STAGE 2 — RecursiveCharacterTextSplitter:
        Some markdown sections are still too long after header splitting.
        Stage 2 handles those using the same character-based strategy as
        _split_recursive(), preserving all metadata from Stage 1.

    Example metadata after both stages:
        {
            "doc_id": "uuid-123",
            "filename": "docs.md",
            "page": 1,
            "Header 1": "API Reference",
            "Header 2": "Authentication",
            "chunk_index": 4,
            "start_index": 1200
        }
    """
    # ── Stage 1: split on markdown headers ───────────────────────────────────
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),  # H1
            ("##", "Header 2"),  # H2
            ("###", "Header 3"),  # H3
        ],
        # strip_headers=False keeps the header line in the chunk content.
        # This means the chunk is self-contained — the reader sees the
        # section title along with its content.
        strip_headers=False,
    )

    # ── Stage 2: character splitter for oversized sections ───────────────────
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        # Stage 1: split by headers
        # MarkdownHeaderTextSplitter.split_text() returns List[Document]
        # Each Document has header hierarchy in its metadata.
        header_chunks = header_splitter.split_text(doc.page_content)

        # Merge parent Document's metadata into each header chunk.
        # Priority: parent metadata first, then header metadata on top.
        # This means doc_id, filename, page, etc. are always present,
        # and Header 1/2/3 keys are added on top.
        for hc in header_chunks:
            hc.metadata = {**doc.metadata, **hc.metadata}

        # Stage 2: further split any sections that are still too large
        final_chunks = char_splitter.split_documents(header_chunks)
        all_chunks.extend(final_chunks)

    return all_chunks


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────


def estimate_chunk_count(
    total_chars: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """
    Estimate how many chunks a document will produce before actually splitting.

    Useful for progress bars and pre-flight cost estimates in the UI.

    Formula:
        effective_step = chunk_size - chunk_overlap
        estimated_chunks = ceil(total_chars / effective_step)

    This is an approximation — actual count varies based on where natural
    separators (paragraph breaks, sentences) fall relative to chunk boundaries.

    Args:
        total_chars:   Total character count of the raw document text.
        chunk_size:    Override for settings.chunk_size.
        chunk_overlap: Override for settings.chunk_overlap.

    Returns:
        Estimated number of chunks (always >= 1).

    Example:
        >>> estimate_chunk_count(50_000)   # 50k-char document
        62                                  # approx 62 chunks
    """
    import math

    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    step = max(size - overlap, 1)  # chars advanced per chunk
    return max(1, math.ceil(total_chars / step))
