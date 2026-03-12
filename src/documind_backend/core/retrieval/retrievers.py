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

  Fusion (RRF):       EnsembleRetriever combines Dense + Sparse using
                      Reciprocal Rank Fusion. A chunk that ranks #1 in Dense
                      AND #3 in Sparse beats one that ranks #1 in only one.
                      Weights: [0.6 Dense, 0.4 Sparse] — tune to your corpus.

  Layer 2 — RE-RANKING (optional, Cohere cross-encoder)
  ─────────────────────────────────────────────────────
  Takes the top-K fusion results and re-scores them by running a
  cross-encoder model that reads (question + chunk) TOGETHER.
  Bi-encoders (embeddings) encode question and chunk independently —
  cross-encoders read them jointly and produce much more accurate scores.
  Cost: one extra Cohere API call per query. Reduces top-K to top-N.
  Skipped if COHERE_API_KEY is not set.

  Layer 3 — CONFIDENCE FILTERING
  ─────────────────────────────────────────────────────
  Drops chunks whose relevance score falls below settings.confidence_threshold.
  If ALL chunks are filtered: the chain returns "I don't know" instead of
  hallucinating an answer from irrelevant context. This is the anti-hallucination
  safety valve.

FULL PIPELINE VISUALIZED:
  question
     │
     ├──────────────────────────┐
     ▼                          ▼
  Dense retriever           Sparse retriever
  (Chroma cosine sim)       (BM25 keyword)
     │   top-K chunks           │   top-K chunks
     └──────────┬───────────────┘
                ▼
          EnsembleRetriever
          (RRF fusion → re-ranked combined list)
                │
                ▼ (if COHERE_API_KEY set)
          CohereRerank
          (cross-encoder → top-N most relevant)
                │
                ▼
          confidence filter
          (drop score < threshold)
                │
                ▼
          List[Document]  →  injected into prompt as context
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

    This is the main entry point for retrieval. The rag_chain and agent
    both call this to get a retriever scoped to the right documents.

    Args:
        doc_ids:       If provided, restrict search to ONLY these documents.
                       Pass None to search across ALL ingested documents.
                       Supports one document (exact match) or many ($in filter).
        use_reranking: Apply Cohere cross-encoder re-ranking as a second pass.
                       Requires COHERE_API_KEY in environment.
                       Set False to reduce latency when accuracy is less critical.

    Returns:
        A LangChain BaseRetriever. All retrievers share the same interface:
            docs = retriever.invoke("user question")
            # → List[Document], sorted by relevance descending

    Example:
        >>> # Search all documents
        >>> retriever = build_retriever()

        >>> # Search only two specific documents
        >>> retriever = build_retriever(doc_ids=["uuid-1", "uuid-2"])

        >>> # Fast retrieval without re-ranking (e.g. for the agent's planning step)
        >>> retriever = build_retriever(use_reranking=False)
    """
    # ── Build metadata filter for ChromaDB ───────────────────────────────────
    # ChromaDB WHERE clause syntax:
    #   Single doc:   {"doc_id": "uuid-123"}
    #   Multiple docs: {"doc_id": {"$in": ["uuid-1", "uuid-2"]}}
    #   No filter:    None  (search entire collection)
    chroma_filter = _build_chroma_filter(doc_ids)

    # ── Layer 1a: Dense retriever (ChromaDB vector search) ───────────────────
    dense_retriever = _build_dense_retriever(chroma_filter)

    # ── Layer 1b: Sparse retriever (BM25 keyword search) ─────────────────────
    sparse_retriever = _build_sparse_retriever(doc_ids)

    if sparse_retriever is None:
        # No documents in the store yet — fall back to dense-only
        logger.warning(
            "[retrievers] No documents found for BM25 index — " "using dense-only retrieval"
        )
        base_retriever = dense_retriever
    else:
        # ── Layer 1 fusion: EnsembleRetriever (RRF) ──────────────────────────
        base_retriever = _build_ensemble_retriever(dense_retriever, sparse_retriever)

    # ── Layer 2: Re-ranking (optional) ───────────────────────────────────────
    if use_reranking and settings.cohere_api_key:
        logger.info("[retrievers] Re-ranking enabled (Cohere)")
        return _wrap_with_reranker(base_retriever)

    if use_reranking and not settings.cohere_api_key:
        logger.info(
            "[retrievers] Re-ranking requested but COHERE_API_KEY not set — " "skipping re-ranking"
        )

    return base_retriever


# ══════════════════════════════════════════════════════════════════════════════
# Layer builders (private)
# ══════════════════════════════════════════════════════════════════════════════


def _build_chroma_filter(doc_ids: list[str] | None) -> dict | None:
    """
    Build the ChromaDB WHERE clause from a list of doc_ids.

    ChromaDB uses a MongoDB-style query syntax for metadata filtering.
    Returns None when doc_ids is empty/None to search the full collection.
    """
    if not doc_ids:
        return None

    if len(doc_ids) == 1:
        # Simple equality — faster than $in for a single value
        return {"doc_id": doc_ids[0]}

    # $in operator: match any of the provided values
    return {"doc_id": {"$in": doc_ids}}


def _build_dense_retriever(chroma_filter: dict | None) -> BaseRetriever:
    """
    Build the dense (vector similarity) retriever from ChromaDB.

    search_type="similarity":
        Standard cosine similarity search. Returns the top-k chunks
        whose embedding vectors are closest to the query vector.

    search_type="mmr" (alternative):
        Maximum Marginal Relevance. Trades off relevance vs. diversity —
        avoids returning 5 nearly-identical chunks about the same sentence.
        Use if you find retrieved chunks are too repetitive.

    k=settings.retrieval_top_k:
        How many chunks to return. Default 5. More = more context for the
        LLM but higher cost + more noise. Re-ranking reduces this to top-N.
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
    Build a BM25 sparse retriever from chunks currently in ChromaDB.

    BM25 (Best Match 25) is a probabilistic keyword-ranking function.
    It scores chunks by:
      - Term Frequency (TF): how often the query term appears in the chunk
      - Inverse Document Frequency (IDF): how rare the term is across all chunks
        (rare terms are more informative)
      - Document length normalization: penalizes very long chunks

    WHY WE BUILD IT FROM CHROMA CONTENTS:
    BM25Retriever needs the raw text of all documents to build its index.
    We fetch the texts directly from ChromaDB so we don't need a second
    storage layer. The BM25 index is rebuilt on every call — for production
    with 10,000+ chunks, consider caching it and invalidating on new ingests.

    Returns:
        BM25Retriever instance, or None if no documents are stored yet.
    """
    vectorstore = get_vectorstore()
    collection = vectorstore._collection

    # Fetch texts + metadata from Chroma (no vectors needed)
    where_filter = _build_chroma_filter(doc_ids)
    fetch_kwargs: dict = {"include": ["documents", "metadatas"]}
    if where_filter:
        fetch_kwargs["where"] = where_filter

    results = collection.get(**fetch_kwargs)

    texts = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not texts:
        return None  # collection is empty — caller handles this

    # Reconstruct LangChain Documents from raw Chroma data
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas)]

    logger.debug(f"[retrievers] BM25 index built from {len(docs)} chunks")

    return BM25Retriever.from_documents(
        docs,
        k=settings.retrieval_top_k,
    )


def _build_ensemble_retriever(
    dense: BaseRetriever,
    sparse: BM25Retriever,
) -> EnsembleRetriever:
    """
    Combine dense and sparse retrievers using Reciprocal Rank Fusion (RRF).

    RRF ALGORITHM:
    For each chunk, its score = Σ 1 / (rank_in_retriever_i + k)
    where k=60 is a smoothing constant.

    Example (k=60):
      Chunk A: rank #1 in Dense, rank #3 in Sparse
        Score = 1/(1+60) + 1/(3+60) = 0.0164 + 0.0159 = 0.0323

      Chunk B: rank #1 in Sparse only (not in Dense top-K)
        Score = 0 + 1/(1+60) = 0.0164

    Chunk A wins because it appears in BOTH retrievers.

    WEIGHTS [0.6, 0.4]:
    Dense gets 60% weight, Sparse gets 40%.
    Tune based on your corpus:
      - More keyword-heavy docs (legal, code) → increase sparse weight
      - More semantic/conversational docs (reports, books) → increase dense weight
    """
    return EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.6, 0.4],  # must sum to 1.0
    )


def _wrap_with_reranker(base_retriever: BaseRetriever) -> ContextualCompressionRetriever:
    """
    Wrap any retriever with Cohere's cross-encoder re-ranker.

    HOW CROSS-ENCODING DIFFERS FROM BI-ENCODING:

    Bi-encoder (what embeddings do):
      question  →  [embed]  →  q_vector
      chunk     →  [embed]  →  c_vector
      score = cosine(q_vector, c_vector)
      The question and chunk NEVER interact during scoring.

    Cross-encoder (what Cohere Rerank does):
      [question + chunk]  →  [transformer]  →  relevance_score
      The model reads BOTH at the same time — full attention between
      question tokens and chunk tokens. Far more accurate but can't be
      pre-computed (must run at query time).

    ContextualCompressionRetriever workflow:
      1. Calls base_retriever.invoke(question)  → top-K chunks (e.g. 10)
      2. Sends all K (question, chunk) pairs to Cohere Rerank API
      3. Cohere returns relevance_score for each pair
      4. Keeps only top-N by score (settings.rerank_top_n, default 3)

    The final N chunks are higher-quality than the original K — less noise
    injected into the prompt.
    """
    from langchain_cohere import CohereRerank

    compressor = CohereRerank(
        cohere_api_key=settings.cohere_api_key,
        top_n=settings.rerank_top_n,
        model="rerank-english-v3.0",
        # rerank-multilingual-v3.0  → use for non-English documents
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
    Remove chunks whose relevance score is below the confidence threshold.

    This is the anti-hallucination gate. When the retriever finds no
    relevant context (all chunks score below threshold), we return
    has_relevant_context=False and the chain returns "I don't know"
    rather than fabricating an answer.

    RELEVANCE SCORES:
    After Cohere re-ranking: scores are 0.0–1.0 (probability of relevance).
    After embedding-only retrieval: scores are cosine distances (0.0–1.0).
    For BM25-only: scores are BM25 term frequencies (not normalized to 0-1).

    For chunks without a relevance_score in metadata (e.g. from BM25 only),
    we default to 1.0 (pass through) — BM25 doesn't produce normalized scores.

    Args:
        docs:      Retrieved documents. May have 'relevance_score' in metadata.
        threshold: Minimum score to keep. Defaults to settings.confidence_threshold.
                   Tune this:
                     0.1 → very permissive (more context, more noise)
                     0.5 → balanced
                     0.7 → strict (less noise, might miss edge-case relevant chunks)

    Returns:
        Tuple of:
          - filtered_docs: only chunks that passed the threshold
          - has_relevant_context: False if ALL chunks were filtered out

    Example:
        >>> docs = retriever.invoke("What is the refund policy?")
        >>> filtered, has_context = filter_by_confidence(docs)
        >>> if not has_context:
        ...     return "I don't have enough information to answer."
    """
    min_score = threshold if threshold is not None else settings.confidence_threshold

    filtered = []
    for doc in docs:
        # Cohere Rerank injects 'relevance_score' into metadata.
        # ChromaDB similarity search doesn't — so default to 1.0 (pass through).
        score = float(doc.metadata.get("relevance_score", 1.0))

        if score >= min_score:
            filtered.append(doc)
        else:
            logger.debug(
                f"[retrievers] Filtered chunk below threshold "
                f"(score={score:.3f} < {min_score}) | "
                f"file={doc.metadata.get('filename', '?')} | "
                f"page={doc.metadata.get('page', '?')}"
            )

    has_context = len(filtered) > 0

    if not has_context:
        logger.info(
            f"[retrievers] All {len(docs)} chunks scored below "
            f"confidence threshold ({min_score}). "
            "Chain will return 'I don't know' response."
        )

    return filtered, has_context


# ══════════════════════════════════════════════════════════════════════════════
# Utility — used by tests and the agent's planning step
# ══════════════════════════════════════════════════════════════════════════════


def retrieve_raw(
    question: str,
    doc_ids: list[str] | None = None,
    use_reranking: bool = True,
) -> list[Document]:
    """
    Convenience wrapper: build retriever and invoke it in one call.

    Returns the raw retrieved documents WITHOUT confidence filtering.
    Use this when you need to inspect scores or do your own filtering.

    For the standard query path, prefer calling build_retriever() directly
    so you can control the retriever lifecycle.

    Args:
        question:      The user's question to retrieve context for.
        doc_ids:       Optional document scope.
        use_reranking: Apply Cohere re-ranking.

    Returns:
        List of Documents sorted by relevance descending.
    """
    retriever = build_retriever(doc_ids=doc_ids, use_reranking=use_reranking)
    docs = retriever.invoke(question)

    logger.info(
        f"[retrievers] retrieve_raw → {len(docs)} chunks " f"for question='{question[:60]}...'"
    )
    return docs
