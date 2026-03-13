"""
core/chains/rag_chain.py
-------------------------
The RAG (Retrieval-Augmented Generation) chain — the core of DocuMind.

WHAT RAG DOES IN THREE WORDS: Retrieve → Augment → Generate

  Retrieve:  Find the most relevant chunks from ChromaDB for the question.
  Augment:   Inject those chunks as context into the prompt.
  Generate:  Ask llama3.2 to answer ONLY from that context.

WHY RAG OVER PLAIN llama3.2:
  llama3.2 was trained on general internet data.
  It has never seen YOUR uploaded documents.
  RAG gives it the specific context it needs to answer accurately.
  Without RAG: llama3.2 guesses → hallucination.
  With RAG:    llama3.2 reads → grounded answer.

LCEL PIPELINE (LangChain Expression Language):

  question ──► RunnableParallel ──► prompt ──► llama3.2 ──► StrOutputParser
                   │         │
                   ▼         ▼
              retriever   passthrough
              (fetch        (keep
              context)     question)

  The | pipe operator chains Runnables lazily.
  Nothing executes until .invoke() or .astream() is called.
  Every component implements the same interface:
    .invoke(input)   → synchronous single call
    .astream(input)  → async generator, yields tokens one by one
    .batch(inputs)   → parallel execution across multiple inputs

STREAMING:
  .astream() on the chain yields tokens as llama3.2 generates them.
  The FastAPI SSE route wraps these tokens in "data: ...\n\n" format
  and streams them to the browser. User sees text appear word by word.

ANTI-HALLUCINATION:
  If no retrieved chunks pass the confidence threshold, the chain
  short-circuits and returns "I don't know" WITHOUT calling llama3.2.
  This saves inference time AND prevents fabricated answers.
"""

import logging
from typing import AsyncIterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from documind_backend.config import settings
from documind_backend.core.retrieval.retrievers import (
    build_retriever,
    filter_by_confidence,
)
from documind_backend.models.schemas import QueryResponse, SourceChunk

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Prompts
# ══════════════════════════════════════════════════════════════════════════════

# System prompt — the most important piece of the whole application.
# Every instruction here directly impacts answer quality and honesty.
#
# KEY RULES EXPLAINED:
#   Rule 1 — "ONLY the context": prevents hallucination from training data.
#             llama3.2 knows a lot — we must explicitly forbid it from
#             using that knowledge and force it to cite the documents.
#   Rule 2 — "I don't have enough information": honest uncertainty is better
#             than a confident wrong answer. Users trust the system more
#             when it admits what it doesn't know.
#   Rule 3 — Reference excerpts: builds user trust + allows verification.
#   Rule 4 — Be concise: llama3.2 tends to pad answers. This reins it in.
SYSTEM_PROMPT = """You are DocuMind, a precise and honest document assistant.

Your ONLY job is to answer questions using the document excerpts provided below.

STRICT RULES — follow these exactly:
1. Use ONLY the provided context excerpts to form your answer.
   Do NOT use your general knowledge or training data.
2. If the context does not contain enough information to answer, respond with:
   "I don't have enough information in the provided documents to answer this."
   Do not attempt to guess or fill gaps with outside knowledge.
3. When you use specific information, reference where it came from naturally:
   "According to page 3...", "The document states...", "In section 2.1..."
4. Be concise and direct. Do not pad your answer with unnecessary phrases.
5. If the question is completely unrelated to the documents, say so politely.

CONTEXT EXCERPTS:
{context}"""

HUMAN_PROMPT = "{question}"

# The "I don't know" response — returned when confidence filtering
# removes ALL retrieved chunks (nothing relevant found).
# Returned as-is, without calling llama3.2, saving inference time.
NO_CONTEXT_RESPONSE = (
    "I don't have enough information in the provided documents "
    "to answer this question."
)


# ══════════════════════════════════════════════════════════════════════════════
# Context formatting
# ══════════════════════════════════════════════════════════════════════════════

def _format_context(docs: list[Document]) -> str:
    """
    Convert retrieved Document chunks into a single formatted context string
    that gets injected into the {context} slot of the system prompt.

    FORMAT:
      --- Excerpt 1 (report.pdf, page 3) ---
      [chunk text here]

      --- Excerpt 2 (contract.docx, page 7) ---
      [chunk text here]

    WHY LABEL EACH EXCERPT:
      The labels ("report.pdf, page 3") let llama3.2 cite sources
      naturally in its answer. Without labels, it can't say where
      information came from — just that it exists in "the documents".

    Args:
        docs: Retrieved and filtered Document chunks.

    Returns:
        Formatted multi-excerpt string, or fallback message if docs is empty.
    """
    if not docs:
        return "No relevant context found in the documents."

    parts = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "unknown document")
        page     = doc.metadata.get("page", "?")
        parts.append(
            f"--- Excerpt {i} ({filename}, page {page}) ---\n"
            f"{doc.page_content.strip()}"
        )

    # Join with double newline so llama3.2 sees clear visual separation
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# LLM factory
# ══════════════════════════════════════════════════════════════════════════════

def _get_llm(streaming: bool = False) -> ChatOllama:
    """
    Return a ChatOllama instance pointing at your local Ollama server.

    Args:
        streaming: True enables token-by-token streaming via .astream().
                   False returns the complete response in one call via .invoke().

    OLLAMA PARAMETERS EXPLAINED:
      temperature=0:
        Deterministic output — same question always gets same answer.
        Higher temperature (0.7+) = more creative but less reliable.
        For document Q&A, determinism is what you want.

      num_ctx=4096:
        Context window size in tokens.
        llama3.2 supports up to 128k, but 4096 is enough for 3-5 chunks
        and keeps inference fast on your 8GB Mac.
        Increase to 8192 if you find answers getting cut off.

      num_predict=1024:
        Max tokens to generate in the response.
        1024 ≈ ~750 words — enough for detailed answers.
        Prevents runaway generation on complex questions.
    """
    return ChatOllama(
        model=settings.ollama_chat_model,      # "llama3.2:latest"
        base_url=settings.ollama_base_url,     # "http://localhost:11434"
        temperature=0,
        num_ctx=4096,
        num_predict=1024,
        # streaming is handled natively by Ollama — no extra config needed
    )


# ══════════════════════════════════════════════════════════════════════════════
# Chain builder
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_chain(
    doc_ids: list[str] | None = None,
    use_reranking: bool = True,
    streaming: bool = False,
):
    """
    Build the full RAG chain as an LCEL pipeline.

    This is a factory — call it to get a chain scoped to specific documents.
    The chain itself is lazy: it doesn't execute until .invoke() or .astream().

    CHAIN ANATOMY:
      setup = RunnableParallel({
          "context":  retriever | format_fn,   ← runs retrieval
          "question": RunnablePassthrough(),    ← passes question through unchanged
      })
      chain = setup | prompt | llm | parser

    DATA FLOW:
      Input:  {"question": "What are the payment terms?"}
                           │
                           ▼
      RunnableParallel (both branches run concurrently):
        context branch:  retriever.invoke(question) → [Doc1, Doc2, Doc3]
                         → _format_context([...])   → "--- Excerpt 1 ---\n..."
        question branch: passthrough                → "What are the payment terms?"
                           │
                           ▼
      Merged: {"context": "--- Excerpt 1 ---\n...", "question": "What are..."}
                           │
                           ▼
      Prompt: ChatPromptTemplate fills {context} and {question} slots
                           │
                           ▼
      llama3.2: reads prompt, generates answer tokens
                           │
                           ▼
      StrOutputParser: extracts .content string from AIMessage object

    Args:
        doc_ids:       Scope retrieval to specific documents. None = all docs.
        use_reranking: Apply FlashRank re-ranking (recommended True).
        streaming:     True for SSE streaming endpoint, False for single response.

    Returns:
        LCEL chain Runnable. Use:
          chain.invoke({"question": "..."})        → str
          await chain.astream({"question": "..."}) → AsyncIterator[str]
    """
    llm      = _get_llm(streaming=streaming)
    prompt   = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])
    retriever = build_retriever(doc_ids=doc_ids, use_reranking=use_reranking)

    # RunnableParallel: executes both branches with the same input simultaneously.
    # "context"  branch: retriever finds chunks → format_context renders them
    # "question" branch: RunnablePassthrough() returns the input unchanged
    # Both outputs are merged into {"context": "...", "question": "..."}
    # which is exactly what the prompt template expects.
    setup = RunnableParallel({
        "context":  retriever | RunnableLambda(_format_context),
        "question": RunnablePassthrough(),
    })

    # Full pipeline — the | operator creates a RunnableSequence
    # Each component's output becomes the next component's input
    chain = setup | prompt | llm | StrOutputParser()

    return chain


# ══════════════════════════════════════════════════════════════════════════════
# Non-streaming query — returns full QueryResponse
# ══════════════════════════════════════════════════════════════════════════════

async def run_rag_query(
    question: str,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> QueryResponse:
    """
    Execute a complete RAG query and return a structured response.

    This is the NON-STREAMING path used by POST /query.
    It retrieves chunks, runs confidence filtering, calls llama3.2,
    and packages everything into a QueryResponse with source citations.

    FLOW:
      1. Retrieve chunks (hybrid dense+sparse → FlashRank re-rank)
      2. Confidence filter  → if nothing passes: return NO_CONTEXT_RESPONSE
      3. Format context     → inject chunks into prompt
      4. Call llama3.2      → get complete answer string
      5. Build citations    → attach source info for UI citation panel
      6. Return QueryResponse

    NOTE ON RETRIEVAL:
      We run retrieval manually here (not through the chain) so we can:
        a) Apply confidence filtering before calling llama3.2
        b) Build SourceChunk citations from the actual retrieved docs
      The chain's internal retriever and this retrieval use the same
      build_retriever() call — same results, same config.

    Args:
        question: User's natural language question.
        doc_ids:  Restrict search to these document UUIDs. None = all docs.
        top_k:    Number of source citations to include in response.

    Returns:
        QueryResponse with answer, sources, and has_relevant_context flag.
    """
    logger.info(f"[rag_chain] RAG query: '{question[:80]}...'")

    # ── Step 1: Retrieve ─────────────────────────────────────────────────────
    retriever     = build_retriever(doc_ids=doc_ids, use_reranking=True)
    retrieved_docs = retriever.invoke(question)

    logger.info(f"[rag_chain] Retrieved {len(retrieved_docs)} chunks")

    # ── Step 2: Confidence filter ────────────────────────────────────────────
    # If ALL chunks score below threshold → honest "I don't know"
    # No llama3.2 call → saves ~2-5 seconds of inference time
    filtered_docs, has_context = filter_by_confidence(retrieved_docs)

    if not has_context:
        logger.info("[rag_chain] No confident context — returning fallback response")
        return QueryResponse(
            answer=NO_CONTEXT_RESPONSE,
            sources=[],
            has_relevant_context=False,
        )

    # ── Step 3: Format context ───────────────────────────────────────────────
    context_str = _format_context(filtered_docs)

    # ── Step 4: Call llama3.2 ────────────────────────────────────────────────
    # We build a minimal chain here (no retriever — we already have context)
    # to avoid a second retrieval call.
    llm    = _get_llm(streaming=False)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])

    # Directly format the prompt and invoke — skip the retriever step
    formatted = prompt.invoke({
        "context":  context_str,
        "question": question,
    })
    response = await llm.ainvoke(formatted)
    answer   = response.content

    logger.info(f"[rag_chain] Answer generated ({len(answer)} chars)")

    # ── Step 5: Build source citations ───────────────────────────────────────
    # SourceChunk objects are shown in the UI citation panel.
    # User can click each one to see the exact text the answer came from.
    n_citations = top_k or settings.rerank_top_n
    sources = [
        SourceChunk(
            doc_id=doc.metadata.get("doc_id", ""),
            filename=doc.metadata.get("filename", ""),
            page=doc.metadata.get("page"),
            chunk_index=doc.metadata.get("chunk_index", i),
            # Truncate content for display — full text is too long for a card
            content=doc.page_content[:500].strip(),
            relevance_score=float(doc.metadata.get("relevance_score", 1.0)),
        )
        for i, doc in enumerate(filtered_docs[:n_citations])
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        has_relevant_context=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Streaming query — yields tokens for SSE
# ══════════════════════════════════════════════════════════════════════════════

async def stream_rag_query(
    question: str,
    doc_ids: list[str] | None = None,
) -> AsyncIterator[str]:
    """
    Stream RAG answer tokens as llama3.2 generates them.

    This is the STREAMING path used by POST /query/stream.
    Yields string tokens one at a time. The FastAPI route wraps each
    token in SSE format ("data: token\n\n") and sends it to the browser.
    The user sees the answer appear word by word — same UX as ChatGPT.

    FLOW:
      1. Retrieve + confidence filter (same as non-streaming)
      2. If no context → yield single "I don't know" string, return
      3. Stream llama3.2 tokens via .astream() → yield each token

    IMPORTANT:
      This is an async generator (uses `yield`, not `return`).
      The FastAPI route must consume it with `async for token in stream_rag_query(...)`.
      Each yielded value is a small string: a word, punctuation, or whitespace.

    Args:
        question: User's question.
        doc_ids:  Restrict search to specific documents.

    Yields:
        String tokens — individual words/punctuation from llama3.2's output.

    Example FastAPI usage:
        async def token_generator():
            async for token in stream_rag_query(question, doc_ids):
                yield f"data: {token}\n\n"
        return StreamingResponse(token_generator(), media_type="text/event-stream")
    """
    logger.info(f"[rag_chain] Streaming RAG query: '{question[:80]}...'")

    # ── Step 1: Retrieve + filter ────────────────────────────────────────────
    retriever     = build_retriever(doc_ids=doc_ids, use_reranking=True)
    retrieved_docs = retriever.invoke(question)
    filtered_docs, has_context = filter_by_confidence(retrieved_docs)

    # ── Step 2: Short-circuit if no relevant context ─────────────────────────
    if not has_context:
        logger.info("[rag_chain] No confident context — streaming fallback response")
        yield NO_CONTEXT_RESPONSE
        return   # stops the generator — no llama3.2 call

    # ── Step 3: Build prompt with pre-retrieved context ──────────────────────
    context_str = _format_context(filtered_docs)
    prompt      = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])
    formatted = prompt.invoke({
        "context":  context_str,
        "question": question,
    })

    # ── Step 4: Stream tokens from llama3.2 ──────────────────────────────────
    # ChatOllama.astream() yields AIMessageChunk objects.
    # Each chunk has a .content attribute — the actual token string.
    # StrOutputParser is not used here because we yield manually.
    llm = _get_llm(streaming=True)
    async for chunk in llm.astream(formatted):
        token = chunk.content
        if token:   # skip empty chunks (Ollama sometimes sends empty strings)
            yield token