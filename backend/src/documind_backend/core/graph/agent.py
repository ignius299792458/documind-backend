"""
core/graph/agent.py
--------------------
LangGraph-powered agent for complex multi-step document queries.

WHEN TO USE THE AGENT vs THE SIMPLE RAG CHAIN:

  Simple RAG chain (rag_chain.py):
    question → one retrieval → one LLM call → answer
    Use for: 90% of queries. Fast, cheap, deterministic.

  Agent (this file):
    question → classify → plan → retrieve × N → reason → answer
    Use for:
      - Multi-hop questions: "Compare section 3 and section 7 on pricing"
      - Questions requiring multiple searches to answer fully
      - Ambiguous questions that need clarification through iteration
      - Cross-document comparisons: "How do doc A and doc B differ on X?"

LANGGRAPH CONCEPTS USED:

  StateGraph:
    The agent is a directed graph. Each node is a Python function.
    State flows between nodes as a TypedDict. Nodes read state,
    do work, and return partial state updates (merged automatically).

  State (TypedDict with Annotated):
    Annotated[list, operator.add] means when a node returns
    {"messages": [new_msg]}, LangGraph APPENDS to the existing list
    rather than replacing it. This is how conversation history builds up.

  Nodes:
    classify_node → plan_node → retrieve_node (loop) → reason_node → respond_node

  Conditional edges:
    After retrieve_node, a routing function decides:
      - still items in plan? → loop back to retrieve_node
      - plan exhausted?      → go to reason_node
    This is how the agent loops without Python recursion.

  MemorySaver (checkpointer):
    Snapshots the full graph state after every node.
    Enables resumability: if the process crashes mid-run,
    restart with the same thread_id and it picks up where it stopped.
    In production replace with SqliteSaver or PostgresSaver.

AGENT FLOW DIAGRAM:

  START
    │
    ▼
  classify_node
  (simple query vs multi-step?)
    │
    ▼
  plan_node
  (if multi-step: decompose into sub-questions)
    │
    ▼
  retrieve_node ◄─────────────────┐
  (retrieve for next plan item)   │
    │                             │
    ▼                             │
  [more plan items?]──── YES ─────┘
    │
    NO
    ▼
  reason_node
  (synthesize answer from ALL retrieved context)
    │
    ▼
  respond_node
  (format final answer)
    │
    ▼
   END
"""

import logging
import operator
from typing import Annotated, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from documind_backend.config import settings
from documind_backend.core.retrieval.retrievers import build_retriever, filter_by_confidence
from documind_backend.core.chains.rag_chain import (
    _format_context,
    SYSTEM_PROMPT,
    NO_CONTEXT_RESPONSE,
)

logger = logging.getLogger(__name__)

# Safety limit — prevents infinite retrieval loops if planning goes wrong
MAX_ITERATIONS = 4


# ══════════════════════════════════════════════════════════════════════════════
# Agent State
# ══════════════════════════════════════════════════════════════════════════════


class AgentState(TypedDict):
    """
    The shared mutable state that flows through every node in the graph.

    LangGraph merges node return values into this state automatically.
    A node returns a dict with ONLY the keys it wants to update —
    it does not need to return the full state.

    ANNOTATION EXPLAINED:
      Annotated[list[BaseMessage], operator.add]
        → when a node returns {"messages": [new_msg]},
          LangGraph calls: existing_messages + [new_msg]
          This APPENDS rather than REPLACES — how history accumulates.

      All other fields (str, list[str], etc.) are REPLACED on update:
        node returns {"question": "new question"} → replaces old value
    """

    # Full conversation message history (HumanMessage + AIMessage)
    # operator.add = append semantics (new messages join the list)
    messages: Annotated[list[BaseMessage], operator.add]

    # The user's original question — never mutated after classify_node
    question: str

    # If the question needs to be searched within specific documents only.
    # None = search all documents in the collection.
    doc_ids: list[str] | None

    # The agent's retrieval plan — list of sub-questions to answer in order.
    # Simple query: ["original question"]
    # Multi-step: ["sub-question 1", "sub-question 2", "sub-question 3"]
    # retrieve_node pops items off the front of this list.
    plan: list[str]

    # All Document chunks retrieved across ALL retrieval steps, accumulated.
    # Each item is {"page_content": "...", "metadata": {...}}
    # We serialize to dicts (not Document objects) because TypedDict state
    # must be JSON-serializable for the MemorySaver checkpointer.
    retrieved_docs: list[dict]

    # True when at least one retrieved chunk passed confidence filtering.
    # False triggers the "I don't know" response in reason_node.
    has_context: bool

    # The final answer string returned to the user.
    final_answer: str

    # How many retrieve_node calls have happened.
    # Compared against MAX_ITERATIONS in the routing function
    # as a safety circuit breaker.
    iteration_count: int


# ══════════════════════════════════════════════════════════════════════════════
# LLM factory
# ══════════════════════════════════════════════════════════════════════════════


def _get_llm() -> ChatOllama:
    """
    Return ChatOllama for the agent's internal reasoning calls.

    The agent makes several LLM calls:
      - classify_node: classify query type
      - plan_node:     decompose into sub-questions
      - reason_node:   synthesize final answer

    All use temperature=0 for deterministic, reliable behavior.
    num_ctx=4096 is enough for agent reasoning prompts.
    """
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_creativity_temperature,
        num_ctx=4096,
        num_predict=1024,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Node functions
# ══════════════════════════════════════════════════════════════════════════════
# Each node receives the FULL current state and returns a PARTIAL update.
# LangGraph merges the partial update into the state before the next node.


def classify_node(state: AgentState) -> dict:
    """
    Node 1 — Classify query complexity.

    Decides whether the question needs:
      "simple"     → single retrieval is enough → plan = [original_question]
      "multi-step" → multiple targeted searches → go to plan_node

    HOW THE CLASSIFICATION WORKS:
      We ask llama3.2 to classify with a strict prompt.
      The response is parsed — if it contains "multi" we treat it as
      multi-step, otherwise simple. Robust to minor wording variations.

    Returns partial state update with 'plan' populated.
    """
    llm = _get_llm()
    question = state["question"]

    logger.info(f"[agent] classify_node | question='{question[:60]}...'")

    classification_prompt = f"""Analyze this question and respond with ONLY one word.

Question: "{question}"

Does answering this require:
A) ONE search — a single specific lookup is sufficient
B) MULTIPLE searches — requires searching for different aspects separately,
   or comparing different sections, or building up an answer from parts

Respond with ONLY the word "simple" or "multi-step". Nothing else."""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    raw = str(response.content).strip().lower()
    query_type = "multi-step" if "multi" in raw else "simple"

    logger.info(f"[agent] Query classified as: {query_type}")

    if query_type == "simple":
        # Simple query: plan is just the original question
        # Goes straight to retrieve_node → reason_node
        return {
            "plan": [question],
            "messages": [
                AIMessage(content="Query type: simple. Proceeding with direct retrieval.")
            ],
        }

    # Multi-step: delegate decomposition to plan_node
    # Return empty plan — plan_node will fill it
    return {
        "plan": [],
        "messages": [AIMessage(content="Query type: multi-step. Building retrieval plan.")],
    }


def plan_node(state: AgentState) -> dict:
    """
    Node 2 — Decompose a complex question into targeted sub-questions.

    ONLY called when classify_node returns "multi-step" (plan is empty).
    For simple queries, plan is already set and this node is skipped
    via the conditional edge routing.

    WHY DECOMPOSE:
      "Compare the termination clauses in section 3 and section 7"
      → needs TWO separate retrievals:
        Sub-question 1: "What does section 3 say about termination?"
        Sub-question 2: "What does section 7 say about termination?"
      → reason_node then synthesizes both into a comparison answer

    DECOMPOSITION PROMPT:
      We tell llama3.2 to produce 2-4 focused sub-questions.
      Each should be answerable with a single document search.
      We parse the numbered list into a Python list.
    """
    llm = _get_llm()
    question = state["question"]

    logger.info(f"[agent] plan_node | decomposing: '{question[:60]}...'")

    plan_prompt = f"""Break this complex question into 2-4 specific sub-questions.
Each sub-question should be answerable with a single focused document search.
Keep sub-questions concrete and specific.

Original question: "{question}"

Return ONLY a numbered list, one sub-question per line. Example format:
1. What does the document say about X?
2. What are the specific terms for Y?
3. How does Z apply to this case?

Your numbered list:"""

    response = llm.invoke([HumanMessage(content=plan_prompt)])

    # Parse "1. Sub-question text" → ["Sub-question text", ...]
    plan = []
    for line in str(response.content).strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit() and ". " in line:
            # Strip the "1. " prefix
            sub_q = line.split(". ", 1)[1].strip()
            if sub_q:
                plan.append(sub_q)

    # Fallback: if parsing failed, use the original question
    if not plan:
        logger.warning("[agent] plan_node failed to parse sub-questions — using original question")
        plan = [question]

    logger.info(f"[agent] Plan created: {plan}")

    return {
        "plan": plan,
        "messages": [AIMessage(content=f"Retrieval plan: {plan}")],
    }


def retrieve_node(state: AgentState) -> dict:
    """
    Node 3 — Execute one retrieval step from the plan.

    Called once per sub-question in the plan (loops via conditional edge).
    On each call:
      1. Pops the FIRST item from plan (the next sub-question to search)
      2. Retrieves relevant chunks from ChromaDB
      3. Filters by confidence
      4. Accumulates results in retrieved_docs
      5. Returns updated plan (one item shorter) and retrieved_docs

    DEDUPLICATION:
      Multiple retrieval steps often find overlapping chunks.
      _deduplicate_docs() removes exact duplicates before accumulating,
      so the same chunk doesn't appear twice in the prompt context.

    SERIALIZATION:
      LangGraph's MemorySaver needs JSON-serializable state.
      Document objects can't be serialized — we convert to dicts:
      {"page_content": "...", "metadata": {"doc_id": "...", ...}}
    """
    plan = state.get("plan", [])
    doc_ids = state.get("doc_ids")

    if not plan:
        # Plan exhausted — this shouldn't happen due to routing,
        # but handle gracefully just in case
        logger.warning("[agent] retrieve_node called with empty plan")
        return {"has_context": len(state.get("retrieved_docs", [])) > 0}

    # Pop the next sub-question
    current_question = plan[0]
    remaining_plan = plan[1:]
    iteration = state.get("iteration_count", 0) + 1

    logger.info(
        f"[agent] retrieve_node | iteration {iteration}/{MAX_ITERATIONS} | "
        f"searching: '{current_question[:60]}...'"
    )

    # Retrieve — use faster no-reranking for speed in multi-step loops.
    # FlashRank re-ranking happens once in reason_node via the full context.
    retriever = build_retriever(doc_ids=doc_ids, use_reranking=True)
    docs = retriever.invoke(current_question)
    filtered, _ = filter_by_confidence(docs)

    # Serialize Documents to dicts for state storage
    new_docs = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in filtered]

    # Merge with previously retrieved docs, removing duplicates
    existing_docs = state.get("retrieved_docs", [])
    all_docs = _deduplicate_docs(existing_docs + new_docs)

    logger.info(
        f"[agent] retrieve_node | found {len(filtered)} chunks | "
        f"total accumulated: {len(all_docs)}"
    )

    return {
        "plan": remaining_plan,  # one item shorter
        "retrieved_docs": all_docs,  # accumulated unique chunks
        "has_context": len(all_docs) > 0,
        "iteration_count": iteration,
    }


def reason_node(state: AgentState) -> dict:
    """
    Node 4 — Synthesize the final answer from all retrieved context.

    Called ONCE after all retrieval steps are complete.
    At this point, retrieved_docs contains chunks from ALL sub-questions.

    SYNTHESIS PROMPT:
      We give llama3.2 the accumulated context from all retrievals
      and the ORIGINAL question (not the sub-questions).
      llama3.2 must synthesize a coherent answer that addresses
      the original question by combining insights across all excerpts.

    CONTEXT LIMIT:
      All accumulated chunks are injected into context.
      For very long multi-step queries this can be large.
      num_ctx=4096 handles up to ~3000 chars of context comfortably.
      Increase to 8192 if you hit truncation on complex queries.
    """
    question = state["question"]
    retrieved_docs = state.get("retrieved_docs", [])
    has_context = state.get("has_context", False)

    logger.info(f"[agent] reason_node | synthesizing from {len(retrieved_docs)} chunks")

    # No relevant context found across ALL retrieval steps
    if not has_context or not retrieved_docs:
        return {
            "final_answer": NO_CONTEXT_RESPONSE,
            "messages": [AIMessage(content=NO_CONTEXT_RESPONSE)],
        }

    # Reconstruct Document objects from serialized dicts for formatting
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=d["page_content"],
            metadata=d["metadata"],
        )
        for d in retrieved_docs
    ]

    context = _format_context(docs)

    # Use the same system prompt as rag_chain.py for consistency
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    answer = response.content

    logger.info(f"[agent] reason_node | answer generated ({len(answer)} chars)")

    return {
        "final_answer": answer,
        "messages": [response],
    }


def respond_node(state: AgentState) -> dict:
    """
    Node 5 — Final formatting / packaging.

    Currently a pass-through — final_answer is already set by reason_node.

    WHY THIS NODE EXISTS:
      It's a deliberate extension point. In the future you can add:
        - Markdown formatting for the response
        - Low-confidence disclaimer if has_context is barely True
        - Logging the Q&A pair to a database for analytics
        - Adding a "sources used" summary at the end of the answer

    Returns the final_answer unchanged (or augmented in the future).
    """
    final_answer = state.get("final_answer", "No answer was generated.")
    logger.info("[agent] respond_node | response ready")
    return {"final_answer": final_answer}


# ══════════════════════════════════════════════════════════════════════════════
# Conditional edge routing functions
# ══════════════════════════════════════════════════════════════════════════════


def route_after_classify(state: AgentState) -> Literal["plan", "retrieve"]:
    """
    After classify_node: decide whether to plan or retrieve directly.

    If plan is empty → classify_node said "multi-step" → go to plan_node
    If plan has items → classify_node said "simple" → go straight to retrieve
    """
    if not state.get("plan"):
        return "plan"  # multi-step: need to decompose first
    return "retrieve"  # simple: go straight to retrieval


def route_after_retrieve(state: AgentState) -> Literal["retrieve", "reason"]:
    """
    After retrieve_node: loop or move to reasoning?

    Continue retrieving if:
      - There are still sub-questions in the plan (plan is not empty)
      - We haven't hit MAX_ITERATIONS (safety circuit breaker)

    Move to reason_node if:
      - Plan is exhausted (all sub-questions have been retrieved)
      - Max iterations reached — prevent infinite loops
    """
    plan = state.get("plan", [])
    iterations = state.get("iteration_count", 0)

    if plan and iterations < MAX_ITERATIONS:
        logger.info(
            f"[agent] routing → retrieve "
            f"({len(plan)} sub-questions remaining, "
            f"iteration {iterations}/{MAX_ITERATIONS})"
        )
        return "retrieve"

    if iterations >= MAX_ITERATIONS:
        logger.warning(
            f"[agent] MAX_ITERATIONS ({MAX_ITERATIONS}) reached — " "forcing move to reason_node"
        )

    return "reason"


# ══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════════════


def build_agent_graph():
    """
    Assemble nodes + edges into a compiled LangGraph application.

    GRAPH STRUCTURE:
      classify → [plan] → retrieve → (loop) → reason → respond → END
                           ↑______________|

    CONDITIONAL EDGES:
      classify → route_after_classify → "plan" or "retrieve"
      retrieve → route_after_retrieve → "retrieve" (loop) or "reason"

    CHECKPOINTER:
      MemorySaver stores full graph state after every node.
      Each run is identified by a thread_id in the config dict.
      Pass the same thread_id to resume an interrupted run.

      Production upgrade path:
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpointer = SqliteSaver.from_conn_string("./agent_state.db")

    Returns:
        Compiled graph app — same Runnable interface as any LangChain component:
          app.invoke(input, config={"configurable": {"thread_id": "..."}})
          await app.astream(input, config={...})
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ───────────────────────────────────────────────────────
    graph.add_node("classify", classify_node)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("respond", respond_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("classify")

    # ── Edges ────────────────────────────────────────────────────────────────

    # classify → plan OR retrieve (conditional)
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "plan": "plan",
            "retrieve": "retrieve",
        },
    )

    # plan → retrieve (always — plan just sets up the sub-questions)
    graph.add_edge("plan", "retrieve")

    # retrieve → retrieve (loop) OR reason (done) (conditional)
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "retrieve": "retrieve",  # loop back for next sub-question
            "reason": "reason",  # all sub-questions retrieved
        },
    )

    # reason → respond → END (always)
    graph.add_edge("reason", "respond")
    graph.add_edge("respond", END)

    # ── Compile ──────────────────────────────────────────────────────────────
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    logger.info("[agent] LangGraph agent compiled successfully")
    return app


# ══════════════════════════════════════════════════════════════════════════════
# Singleton — compiled once, reused per request
# ══════════════════════════════════════════════════════════════════════════════

_agent_app = None


def get_agent():
    """
    Return the compiled agent graph, building it once on first call.

    Using a module-level singleton avoids recompiling the graph on every
    request. The compiled graph is stateless — all per-request state lives
    in the AgentState TypedDict keyed by thread_id.
    """
    global _agent_app
    if _agent_app is None:
        _agent_app = build_agent_graph()
    return _agent_app


# ══════════════════════════════════════════════════════════════════════════════
# Public query function
# ══════════════════════════════════════════════════════════════════════════════


async def run_agent_query(
    question: str,
    doc_ids: list[str] | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Run the agent for a given question and return the final answer.

    This is the entry point called by the FastAPI agent route.

    Args:
        question:   User's natural language question.
        doc_ids:    Restrict retrieval to specific document UUIDs.
        session_id: Thread ID for conversation memory. Pass the same ID
                    across turns to maintain context. Generated if not provided.

    Returns:
        Dict with:
          "answer":          final answer string
          "session_id":      thread_id used (echo back to client)
          "steps_taken":     number of retrieval iterations performed
          "has_context":     whether relevant context was found
          "retrieved_count": total unique chunks accumulated
    """
    import uuid

    # Generate session_id if not provided
    # Same session_id = same MemorySaver checkpoint = conversation continuity
    thread_id = session_id or str(uuid.uuid4())

    # Initial state — nodes will update this as they run
    initial_state: AgentState = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "doc_ids": doc_ids,
        "plan": [],  # filled by classify_node or plan_node
        "retrieved_docs": [],  # accumulated by retrieve_node
        "has_context": False,
        "final_answer": "",
        "iteration_count": 0,
    }

    # LangGraph config — thread_id scopes the checkpointer state
    config = RunnableConfig(configurable={"thread_id": thread_id})

    logger.info(
        f"[agent] run_agent_query | " f"thread_id={thread_id} | " f"question='{question[:60]}...'"
    )

    app = get_agent()

    # ainvoke runs the full graph to completion and returns the final state
    final_state = await app.ainvoke(initial_state, config=config)

    return {
        "answer": final_state.get("final_answer", NO_CONTEXT_RESPONSE),
        "session_id": thread_id,
        "steps_taken": final_state.get("iteration_count", 0),
        "has_context": final_state.get("has_context", False),
        "retrieved_count": len(final_state.get("retrieved_docs", [])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _deduplicate_docs(docs: list[dict]) -> list[dict]:
    """
    Remove duplicate chunks from the accumulated retrieved_docs list.

    Uses the first 200 characters of page_content as a fingerprint.
    Two chunks with the same opening text are considered the same chunk.

    Called in retrieve_node to prevent the same chunk being injected
    into the prompt multiple times across different retrieval steps.
    """
    seen = set()
    unique = []
    for doc in docs:
        # Use first 200 chars as fingerprint — fast and accurate enough
        fingerprint = doc.get("page_content", "")[:200].strip()
        if fingerprint and fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(doc)
    return unique
