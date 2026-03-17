"""
api/routes/query.py
--------------------
Document querying endpoints.

ENDPOINTS:
  POST /query          → non-streaming RAG query, returns full QueryResponse
  POST /query/stream   → streaming RAG query, returns SSE token stream
  POST /agent/query    → LangGraph multi-step agent query

SIMPLE RAG vs AGENT:
  POST /query and POST /query/stream use the simple RAG chain:
    question → retrieve → llama3.2 → answer
    Fast (~3-8s), deterministic, great for 90% of questions.

  POST /agent/query uses the LangGraph agent:
    question → classify → [plan] → retrieve × N → reason → answer
    Slower (multiple LLM calls), better for complex multi-hop questions.
    Examples: "Compare section 3 and section 7", "Summarize all mentions of X"

STREAMING (SSE):
  Server-Sent Events (SSE) is a one-way push protocol from server to browser.
  The browser opens a connection with EventSource and receives a stream of
  "data: token\n\n" messages. Each message is one token from llama3.2.
  The browser appends tokens to the UI in real time — same UX as ChatGPT.

  SSE format:
    data: Hello\n\n
    data:  world\n\n
    data: !\n\n
    data: [DONE]\n\n   ← sentinel to tell the browser the stream is complete

  Why SSE over WebSocket:
    SSE is simpler (HTTP, not WS), auto-reconnects, works through proxies/CDN.
    WebSocket is better for bidirectional communication (we don't need that).
"""

import json
import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

import documind_backend.core.chains.rag_chain as RAGChainService
import documind_backend.core.graph.agent as AgentGraphService
import documind_backend.models.schemas as Schemas

logger = logging.getLogger(__name__)
router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# POST /query — non-streaming RAG
# ══════════════════════════════════════════════════════════════════════════════


@router.post(
    "/query",
    response_model=Schemas.QueryResponse,
    summary="Query documents (non-streaming)",
    description=(
        "Ask a question about your ingested documents. "
        "Returns the complete answer + source citations in one response. "
        "Use POST /query/stream if you want token-by-token streaming."
    ),
)
async def query_documents(request: Schemas.QueryRequest) -> Schemas.QueryResponse:
    """
    Run a RAG query and return the complete answer.

    Non-streaming — waits for llama3.2 to finish generating the full answer
    before returning. Simpler to consume than SSE but higher perceived latency
    (user sees nothing until the full answer is ready, typically 3-8 seconds).

    Use this for:
      - Simple integrations where streaming is hard to implement
      - Programmatic queries where you need the full response as JSON
      - Testing and debugging

    Args:
        request: QueryRequest with question, optional doc_ids, and top_k.

    Returns:
        QueryResponse with:
          answer:               the full answer string from llama3.2
          sources:              list of SourceChunk citations (filename, page, excerpt)
          has_relevant_context: False if nothing was found (honest "I don't know")

    Raises:
        400: Question is empty or too long.
        500: LLM or retrieval failure.
    """
    logger.info(
        f"[query] POST /query | "
        f"question='{request.question[:60]}...' | "
        f"doc_ids={request.doc_ids}"
    )

    try:
        response = await RAGChainService.run_rag_query(
            question=request.question,
            doc_ids=request.doc_ids,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.exception(f"[query] RAG query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {e}",
        )

    logger.info(
        f"[query] Answer ready | "
        f"has_context={response.has_relevant_context} | "
        f"sources={len(response.sources)}"
    )
    return response


# ══════════════════════════════════════════════════════════════════════════════
# POST /query/stream — streaming SSE RAG
# ══════════════════════════════════════════════════════════════════════════════


@router.post(
    "/query/stream",
    summary="Query documents (streaming SSE)",
    description=(
        "Ask a question and receive the answer as a Server-Sent Events stream. "
        "Tokens are sent one by one as llama3.2 generates them. "
        "The response Content-Type is text/event-stream. "
        "Each event is formatted as 'data: <token>\\n\\n'. "
        "The stream ends with 'data: [DONE]\\n\\n'."
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE stream of tokens",
            "content": {
                "text/event-stream": {"example": "data: Hello\n\ndata:  world\n\ndata: [DONE]\n\n"}
            },
        }
    },
)
async def stream_query_documents(request: Schemas.QueryRequest) -> StreamingResponse:
    """
    Stream a RAG answer token by token using Server-Sent Events.

    The frontend opens a connection with fetch() or EventSource and
    receives tokens as they're generated by llama3.2. This gives the
    "typing" effect — much better UX than waiting for the full answer.

    SSE PROTOCOL:
      Each message: "data: <content>\n\n"
      End sentinel: "data: [DONE]\n\n"
      Error event:  "event: error\ndata: <message>\n\n"

    FRONTEND CONSUMPTION EXAMPLE (JavaScript):
      const response = await fetch('/query/stream', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: 'What are the terms?'})
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          const text = decoder.decode(value);
          // text = "data: Hello\n\n" → extract "Hello"
          const token = text.replace('data: ', '').replace('\\n\\n', '');
          if (token !== '[DONE]') appendToUI(token);
      }

    Args:
        request: QueryRequest with question and optional doc_ids.

    Returns:
        StreamingResponse with Content-Type: text/event-stream.
    """
    logger.info(
        f"[query] POST /query/stream | "
        f"question='{request.question[:60]}...' | "
        f"doc_ids={request.doc_ids}"
    )

    async def token_generator():
        """
        Async generator that yields SSE-formatted tokens.

        Wraps stream_rag_query() — which yields raw token strings —
        and formats each one as a proper SSE message.

        On error: yields an SSE error event instead of crashing.
        On completion: yields the [DONE] sentinel so the client knows
        the stream is finished.
        """
        try:
            async for token in RAGChainService.stream_rag_query(
                question=request.question,
                doc_ids=request.doc_ids,
            ):
                # SSE format: "data: <content>\n\n"
                # The double newline is REQUIRED by the SSE spec.
                # Escape newlines in the token to keep SSE format valid.
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"

        except Exception as e:
            # Send an error event to the client instead of silently dying.
            # The client can listen for "event: error" to handle this.
            logger.exception(f"[query] Streaming error: {e}")
            error_msg = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_msg}\n\n"

        finally:
            # Always send [DONE] — even after an error.
            # This tells the client the stream is finished so it doesn't
            # hang waiting for more data.
            yield "data: [DONE]\n\n"
            logger.info("[query] Stream complete")

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            # Disable buffering — some reverse proxies (nginx) buffer SSE by default.
            # These headers force immediate token delivery to the browser.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx-specific: disable buffering
            "Connection": "keep-alive",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST /agent/query — LangGraph multi-step agent
# ══════════════════════════════════════════════════════════════════════════════


@router.post(
    "/agent/query",
    response_model=Schemas.AgentQueryResponse,
    summary="Query documents using the LangGraph agent (multi-step)",
    description=(
        "Use the LangGraph agent for complex questions that require "
        "multiple retrieval steps or cross-document reasoning. "
        "Slower than /query but better for: comparisons, summaries across sections, "
        "multi-hop questions. "
        "Pass session_id to maintain conversation context across turns."
    ),
)
async def agent_query_documents(request: Schemas.AgentQueryRequest) -> Schemas.AgentQueryResponse:
    """
    Run the LangGraph agent for complex multi-step document queries.

    The agent:
      1. Classifies the question (simple vs multi-step)
      2. Plans retrieval sub-questions if needed
      3. Retrieves context iteratively (up to 4 times)
      4. Synthesizes a final answer from all accumulated context

    WHEN TO USE THIS OVER /query:
      ✅ "Compare the pricing in section 3 and section 7"
      ✅ "Summarize all the dates and deadlines mentioned"
      ✅ "What are the differences between the two contract versions?"
      ✅ "Find all clauses related to liability and explain them"

    WHEN TO USE /query INSTEAD:
      ✓  "What is the payment due date?"  (single lookup)
      ✓  "Who signed the contract?"       (single lookup)

    SESSION MEMORY:
      Pass session_id to link this query to a previous conversation.
      The LangGraph MemorySaver checkpointer maintains state per thread_id.
      Without session_id, each call is a fresh isolated conversation.

    Args:
        request: AgentQueryRequest with question, optional doc_ids, session_id.

    Returns:
        AgentQueryResponse with answer, session_id, steps_taken, sources.
    """
    logger.info(
        f"[agent] POST /agent/query | "
        f"question='{request.question[:60]}...' | "
        f"session_id={request.session_id}"
    )

    try:
        result = await AgentGraphService.run_agent_query(
            question=request.question,
            doc_ids=request.doc_ids,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.exception(f"[agent] Agent query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent query failed: {e}",
        )

    logger.info(
        f"[agent] Done | "
        f"steps={result['steps_taken']} | "
        f"has_context={result['has_context']}"
    )

    # Build SourceChunk citations from accumulated retrieved_docs
    # The agent doesn't return sources the same way the RAG chain does —
    # we extract them from the agent's accumulated retrieved_docs.
    sources = _build_agent_sources(result.get("retrieved_docs_raw", []))

    return Schemas.AgentQueryResponse(
        answer=result["answer"],
        session_id=result["session_id"],
        steps_taken=result["steps_taken"],
        sources=sources,
        has_relevant_context=result["has_context"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST /query/batch — batch multiple questions at once
# ══════════════════════════════════════════════════════════════════════════════
@router.post(
    "/query/batch",
    response_model=Schemas.BatchQueryResponse,
    summary="Ask multiple questions in one request",
    description=(
        "Submit up to 10 questions at once. "
        "Questions are processed sequentially (not in parallel) "
        "to avoid overwhelming the local Ollama server. "
        "Returns a list of QueryResponse objects in the same order."
    ),
)
async def batch_query_documents(request: Schemas.BatchQueryRequest) -> Schemas.BatchQueryResponse:
    """
    Process multiple questions sequentially and return all answers.

    Useful for:
      - Pre-fetching answers for a set of known questions
      - Bulk document analysis pipelines
      - Testing retrieval quality across many questions at once

    NOTE: Questions are processed one by one (sequential, not parallel).
    Parallel execution would overwhelm llama3.2 on an 8GB Mac.
    For true parallelism, use multiple separate /query requests.

    Args:
        request: BatchQueryRequest with up to 10 questions.

    Returns:
        BatchQueryResponse with results list in same order as input questions.

    Raises:
        400: More than 10 questions submitted.
    """
    if len(request.questions) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch queries are limited to 10 questions per request.",
        )

    if not request.questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one question is required.",
        )

    logger.info(f"[query] POST /query/batch | {len(request.questions)} question(s)")

    results = []
    for i, question in enumerate(request.questions):
        logger.info(f"[query] Batch question {i+1}/{len(request.questions)}: '{question[:50]}...'")
        try:
            response = await RAGChainService.run_rag_query(
                question=question,
                doc_ids=request.doc_ids,
            )
            results.append(response)
        except Exception as e:
            # Don't fail the whole batch on one error —
            # return an error response for the failed question
            logger.error(f"[query] Batch question {i+1} failed: {e}")
            results.append(
                Schemas.QueryResponse(
                    question=question,
                    answer=f"Query failed: {e}",
                    sources=[],
                    has_relevant_context=False,
                )
            )

    logger.info(f"[query] Batch complete | {len(results)} answers")

    return Schemas.BatchQueryResponse(results=results, total=len(results))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _build_agent_sources(retrieved_docs_raw: list[dict]) -> list[Schemas.SourceChunk]:
    """
    Build SourceChunk citation objects from the agent's raw retrieved_docs.

    The agent stores retrieved docs as serialized dicts (not Document objects)
    in its state. This helper converts them back to SourceChunk Pydantic models
    for the API response.

    Args:
        retrieved_docs_raw: List of {"page_content": "...", "metadata": {...}} dicts.

    Returns:
        List of SourceChunk objects (max 5 for readability in the UI).
    """
    sources = []
    for i, doc in enumerate(retrieved_docs_raw[:5]):  # cap at 5 citations
        meta = doc.get("metadata", {})
        sources.append(
            Schemas.SourceChunk(
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""),
                page=meta.get("page"),
                chunk_index=meta.get("chunk_index", i),
                content=doc.get("page_content", "")[:500].strip(),
                relevance_score=float(meta.get("relevance_score", 1.0)),
            )
        )
    return sources
