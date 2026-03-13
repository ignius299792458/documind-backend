"""
core/retrieval/retrievers.py
-----------------------------
Builds the multi-layer retrieval pipeline that finds the most relevant
chunks for a user's question.

THREE-LAYER RETRIEVAL ARCHITECTURE:

  Layer 1 — HYBRID RETRIEVAL (Dense + Sparse fusion)
  ─────────────────────────────────────────────────────
  Dense  (ChromaDB):  Embeds the question → finds semantically similar chunks.
                      Great for paraphrased / conceptual queries.
                      "What are the payment terms?"  →  finds "invoices due
                      within 30 days" even though those exact words aren't
                      in the question.

  Sparse (BM25):      Classic keyword frequency matching (TF-IDF variant).
                      Great for exact terms, proper nouns, IDs.
                      "Find clause 4.2.1"  →  finds it even if semantically
                      it looks similar to every other clause.

  Fusion (RRF):       EnsembleRetriever combines Dense + Sparse results using
                      ----🤩 RRF: Reciprocal Rank Fusion---. 
                      A chunk that ranks #1 in Dense AND #3 in Sparse beats 
                      one that ranks #1 in only one.
                      Weights: [0.6 Dense, 0.4 Sparse] — tune to your corpus.

  Layer 2 — RE-RANKING (FlashRank cross-encoder, fully local)
  ─────────────────────────────────────────────────────────────
  Takes the top-K fusion results and re-scores them with a cross-encoder
  model (ms-marco-MiniLM-L-12-v2) that reads (question + chunk) TOGETHER.

  WHY CROSS-ENCODER IS MORE ACCURATE THAN BI-ENCODER:
    Bi-encoder (embeddings):
      question → embed → q_vector
      chunk    → embed → c_vector
      score = cosine(q_vector, c_vector)
      Question and chunk are encoded SEPARATELY — they never interact.

    Cross-encoder (FlashRank):
      [question + chunk] → transformer → relevance_score
      The model reads BOTH together with full attention.
      Every question word interacts with every chunk word.
      Catches mismatches that embedding similarity misses.

  FlashRank runs 100% locally — no API, no internet, no cost.
  Model (~50MB) downloads once to ~/.cache/flashrank/ on first use.

  Layer 3 — CONFIDENCE FILTERING
  ─────────────────────────────────────────────────────
  Drops chunks whose relevance score falls below settings.confidence_threshold.
  If ALL chunks are filtered: the chain returns "I don't know" instead of
  hallucinating an answer from irrelevant context. Anti-hallucination gate.

FULL PIPELINE:
              question
                 │
     |───────────├──────────────|
     ▼                          ▼
  Dense retriever           Sparse retriever
  (Chroma + nomic-embed)    (BM25 keyword)
     │   top-K chunks           │   top-K chunks
     └──────────┬───────────────┘
                ▼
          EnsembleRetriever
          (RRF fusion → combined ranked list)
                │
                ▼
          FlashRank cross-encoder
          (re-scores top-K → keeps top-N)
                │
                ▼
          confidence filter
          (drops score < threshold)
                │
                ▼
          List[Document]  →  injected into prompt as context

SETUP REQUIRED:
  poetry add flashrank
  No API keys needed. Everything runs on your local Ollama stack.
"""

import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from documind_backend.config import settings
from documind_backend.core.retrieval.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Public API — the only function most callers need
# ══════════════════════════════════════════════════════════════════════════════

def build_retriever(
    doc_ids: list[str] | None = None,
    use_reranking: bool = True,
) -> BaseRetriever:
    """
    Build and return the full multi-layer retrieval pipeline.

    Called by rag_chain.py and agent.py to get a retriever scoped
    to the right documents. Returns a standard LangChain BaseRetriever
    so callers never need to know which layers are active.

    Args:
        doc_ids:       Restrict search to ONLY these document UUIDs.
                       None = search across ALL ingested documents.
        use_reranking: Apply FlashRank cross-encoder as a second pass.
                       Set False for faster (but less precise) retrieval,
                       e.g. during the agent's planning step where speed
                       matters more than precision.

    Returns:
        BaseRetriever — call with:
            docs = retriever.invoke("your question")
            # → List[Document], sorted by relevance descending

    Examples:
        >>> retriever = build_retriever()
        >>> retriever = build_retriever(doc_ids=["uuid-1", "uuid-2"])
        >>> retriever = build_retriever(use_reranking=False)  # faster
    """
    # ── Build ChromaDB metadata filter ───────────────────────────────────────
    chroma_filter = _build_chroma_filter_query(doc_ids)

    # ── Layer 1a: Dense retriever (ChromaDB + nomic-embed-text vectors) ──────
    dense_retriever = _build_dense_retriever(chroma_filter)

    # ── Layer 1b: Sparse retriever (BM25 keyword matching) ───────────────────
    sparse_retriever = _build_sparse_retriever(doc_ids)

    if sparse_retriever is None:
        # Collection is empty — no documents ingested yet.
        # Fall back to dense-only so queries don't crash on empty DB.
        logger.warning(
            "[retrievers] ChromaDB collection is empty — "
            "using dense-only retrieval (BM25 skipped)"
        )
        base_retriever = dense_retriever
    else:
        # ── Layer 1 fusion: EnsembleRetriever (RRF) ──────────────────────────
        base_retriever = _build_ensemble_retriever(dense_retriever, sparse_retriever)

    # ── Layer 2: FlashRank re-ranking (local cross-encoder) ──────────────────
    # use_reranking arg AND settings.use_reranking must both be True.
    # This lets callers override at call-site while still respecting
    # the global config toggle.
    if use_reranking and settings.use_reranking:
        logger.info("[retrievers] Re-ranking enabled (FlashRank local cross-encoder)")
        return _wrap_with_flashrank(base_retriever)

    logger.info("[retrievers] Re-ranking skipped")
    return base_retriever


# ══════════════════════════════════════════════════════════════════════════════
# Private layer builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_chroma_filter_query(doc_ids: list[str] | None) -> dict | None:
    """
    Build ChromaDB WHERE clause from a list of doc_ids.

    ChromaDB uses MongoDB-style metadata filter syntax:
      Single doc:    {"doc_id": "uuid-123"}
      Multiple docs: {"doc_id": {"$in": ["uuid-1", "uuid-2"]}}
      No filter:     None  (searches entire collection)
    """
    if not doc_ids:
        return None
    if len(doc_ids) == 1:
        return {"doc_id": doc_ids[0]}          # equality is faster than $in
    return {"doc_id": {"$in": doc_ids}}


def _build_dense_retriever(chroma_filter: dict | None) -> BaseRetriever:
    """
    Build the dense (vector similarity) retriever.

    Uses nomic-embed-text vectors stored in ChromaDB.
    At query time, the question is embedded with the same model
    and ChromaDB finds the nearest vectors via cosine similarity.

    search_type options:
      "similarity" — standard cosine similarity (default, recommended)
      "mmr"        — Maximum Marginal Relevance: trades off relevance vs
                     diversity to avoid returning 5 near-identical chunks.
                     Useful if you find retrieved chunks are too repetitive.

    k = settings.retrieval_top_k (default 10):
      Fetch more than you'll use — re-ranking will shrink this to top 3.
      More candidates = re-ranker has better material to work with.
    """
    vectorstore = get_vectorstore()

    search_kwargs: dict = {"k": settings.retrieval_top_k}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def _build_sparse_retriever(doc_ids: list[str] | None) -> BM25Retriever | None:
    """
    Build a BM25 keyword retriever from chunks stored in ChromaDB.

    BM25 (Best Match 25) scores chunks by:
      - Term Frequency (TF):  how often a query term appears in the chunk
      - Inverse Document Frequency (IDF): how rare the term is across all chunks
        (rare/specific terms score higher than common words)
      - Length normalization: longer chunks don't get unfair advantage

    WHY BM25 COMPLEMENTS DENSE RETRIEVAL:
      Dense: "payment terms" → finds "30 days net" (semantic match)
      BM25:  "clause 4.2.1"  → finds exact "4.2.1" (keyword match)
      Together they cover both retrieval modes.

    WHY WE REBUILD FROM CHROMA:
      BM25 needs raw text to build its index. We fetch from ChromaDB
      directly — no second storage needed. Tradeoff: rebuilds on every
      query. For large corpora (10k+ chunks), cache this index and
      invalidate it only when new documents are ingested.

    Returns:
        BM25Retriever, or None if no documents are in the collection yet.
    """
    vectorstore = get_vectorstore()
    collection  = vectorstore._collection

    # Fetch all chunk texts + metadata (skip vectors — not needed for BM25)
    where_filter = _build_chroma_filter_query(doc_ids)
    fetch_kwargs: dict = {"include": ["documents", "metadatas"]}
    if where_filter:
        fetch_kwargs["where"] = where_filter

    results   = collection.get(**fetch_kwargs)
    texts     = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not texts:
        return None  # empty collection — caller handles gracefully

    # Reconstruct LangChain Documents for BM25Retriever
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]

    logger.debug(f"[retrievers] BM25 index built from {len(docs)} chunks")

    return BM25Retriever.from_documents(docs, k=settings.retrieval_top_k)


def _build_ensemble_retriever(
    dense: BaseRetriever,
    sparse: BM25Retriever,
) -> EnsembleRetriever:
    """
    Fuse dense + sparse results using Reciprocal Rank Fusion (RRF).

    RRF SCORING FORMULA:
      score(chunk) = Σ  1 / (rank_in_retriever_i + 60)
                    i

      k=60 is a smoothing constant (standard in IR literature).

    EXAMPLE:
      Chunk A → Dense rank #1, Sparse rank #3
        score = 1/(1+60) + 1/(3+60) = 0.0164 + 0.0159 = 0.0323  ← wins

      Chunk B → Dense rank #1 only (not in Sparse top-K)
        score = 1/(1+60) + 0 = 0.0164

    Chunk A wins because it appears in BOTH — RRF rewards agreement
    between the two retrieval methods.

    WEIGHTS [0.6, 0.4]:
      Dense 60%, Sparse 40% — good default for mixed document types.
      Tune if needed:
        Legal / code docs (keyword-heavy) → try [0.4, 0.6]
        Reports / books (semantic)        → keep [0.6, 0.4] or higher
    """
    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.6, 0.4],   # must sum to 1.0
    )


def _wrap_with_flashrank(base_retriever: BaseRetriever) -> ContextualCompressionRetriever:
    """
    Wrap the retriever with FlashRank local cross-encoder re-ranking.

    WHAT FLASHRANK DOES:
      1. Calls base_retriever.invoke(question)  → top-K chunks (e.g. 10)
      2. For each chunk, scores the pair (question, chunk) together
         using a small BERT-based cross-encoder model
      3. Returns top-N chunks by cross-encoder score (e.g. top 3)

    WHY LOCAL CROSS-ENCODER > EMBEDDING SIMILARITY:
      Embeddings encode question and chunk independently — they can't
      detect subtle mismatches. Cross-encoder reads both together:
        Question: "What are payment terms?"
        Chunk:    "The payment office is on the 3rd floor"
        Embedding similarity: 0.79  (both contain "payment")
        Cross-encoder score:  0.08  (correctly identifies mismatch)

    FLASHRANK MODEL: ms-marco-MiniLM-L-12-v2
      - Trained on MS MARCO passage ranking benchmark
      - ~50MB download, cached at ~/.cache/flashrank/ after first run
      - Runs on CPU — no GPU needed, works on your 8GB Mac
      - Latency: ~50-150ms per query (negligible vs llama3.2 inference)

    OTHER FLASHRANK MODEL OPTIONS (swap via model= parameter):
      "ms-marco-TinyBERT-L-2-v2"  → faster (~20ms), smaller, less accurate
      "ms-marco-MiniLM-L-12-v2"   → balanced (default, recommended)
      "rank-T5-flan"               → most accurate, ~200MB, slower (~300ms)

    IMPORTANT — ContextualCompressionRetriever workflow:
      base_retriever fetches top-K (controlled by retrieval_top_k)
      FlashRank re-scores and keeps top-N (controlled by rerank_top_n)
      Only top-N chunks are passed downstream to the LLM prompt.
    """
    from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

    compressor = FlashrankRerank(
        top_n=settings.rerank_top_n,           # how many to keep (default 3)
        model="ms-marco-MiniLM-L-12-v2",       # local cross-encoder model
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Confidence filtering
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_confidence(
    docs: list[Document],
    threshold: float | None = None,
) -> tuple[list[Document], bool]:
    """
    Drop chunks whose relevance score falls below the confidence threshold.

    This is the anti-hallucination gate. When retrieval finds no relevant
    context, returning has_relevant_context=False tells the chain to reply
    "I don't know" instead of generating a hallucinated answer.

    SCORE SOURCES:
      After FlashRank re-ranking:
        → 'relevance_score' in metadata, range 0.0–1.0
        → 0.0 = irrelevant, 1.0 = perfectly relevant
      After embedding-only (no re-ranking):
        → No 'relevance_score' set by ChromaDB
        → Default to 1.0 (pass all through — no filtering)
      After BM25-only:
        → BM25 scores are not normalized to 0–1
        → Default to 1.0 (pass all through)

    THRESHOLD TUNING (settings.confidence_threshold, default 0.3):
      0.1 → very permissive: more context injected, more noise risk
      0.3 → balanced: good default for general document Q&A
      0.5 → strict: clean context, may miss edge-case relevant chunks
      0.7 → very strict: only extremely relevant chunks pass

    Args:
        docs:      Retrieved documents (may have 'relevance_score' in metadata).
        threshold: Override the default confidence threshold from settings.

    Returns:
        (filtered_docs, has_relevant_context)
        has_relevant_context = False when ALL chunks were filtered out.

    Example:
        >>> docs = retriever.invoke("What is the refund policy?")
        >>> filtered, has_context = filter_by_confidence(docs)
        >>> if not has_context:
        ...     return "I don't have enough information to answer this."
    """
    min_score = threshold if threshold is not None else settings.confidence_threshold

    filtered = []
    for doc in docs:
        # FlashRank injects 'relevance_score' into metadata after re-ranking.
        # ChromaDB similarity search does not set this — default 1.0 passes through.
        score = float(doc.metadata.get("relevance_score", 1.0))

        if score >= min_score:
            filtered.append(doc)
        else:
            logger.debug(
                f"[retrievers] Chunk filtered (score={score:.3f} < {min_score}) | "
                f"file={doc.metadata.get('filename', '?')} | "
                f"page={doc.metadata.get('page', '?')}"
            )

    has_context = len(filtered) > 0

    if not has_context:
        logger.info(
            f"[retrievers] All {len(docs)} chunks scored below "
            f"threshold ({min_score}) — chain will return 'I don't know'"
        )

    return filtered, has_context


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_raw(
    question: str,
    doc_ids: list[str] | None = None,
    use_reranking: bool = True,
) -> list[Document]:
    """
    Convenience wrapper: build retriever + invoke in one call.

    Returns raw retrieved documents WITHOUT confidence filtering.
    Use when you need to inspect scores or handle filtering yourself.

    For the standard query path use build_retriever() directly so you
    control the retriever lifecycle.

    Args:
        question:      User question to retrieve context for.
        doc_ids:       Optional scope to specific documents.
        use_reranking: Apply FlashRank re-ranking (default True).

    Returns:
        List[Document] sorted by relevance descending.
    """
    retriever = build_retriever(doc_ids=doc_ids, use_reranking=use_reranking)
    docs = retriever.invoke(question)

    logger.info(
        f"[retrievers] retrieve_raw → {len(docs)} chunks | "
        f"question='{question[:60]}...'"
    )
    return docs