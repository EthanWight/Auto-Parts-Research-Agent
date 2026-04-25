"""
auto_parts_research_agent.workflow
==================================

Core LangGraph multi-agent workflow for the Auto Parts Research Agent.

This module defines the complete agentic pipeline that accepts a plain-English
vehicle problem description and produces a structured parts/repair recommendation.
The pipeline is built as a directed graph using LangGraph and consists of five
specialist agent nodes that execute in sequence:

    1. **planner_agent**          – Uses the LLM to decompose the problem into
                                    3-5 targeted research steps.
    2. **research_agent**         – Iterates over each plan step, executing a
                                    DuckDuckGo web search and storing raw notes.
    3. **summarizer_agent**       – Synthesises all research notes into a
                                    structured recommendation draft via the LLM.
    4. **human_checkpoint_agent** – Presents the draft to the operator and waits
                                    for explicit approval before proceeding.
    5. **finalize_agent**         – Promotes the approved draft to the official
                                    recommendation field of the workflow state.

Environment variables (loaded automatically from a ``.env`` file):
    OPENAI_API_KEY   : Set to ``"ollama"`` when using a local Ollama server.
    OPENAI_BASE_URL  : Base URL of the OpenAI-compatible API endpoint.
                       Defaults to ``http://localhost:11434/v1`` (Ollama).
    OPENAI_MODEL     : Model identifier understood by the endpoint.
                       Defaults to ``qwen3.5:latest``.
    LANGSMITH_TRACING: When ``"true"``, all agent calls are traced in LangSmith.
    LANGSMITH_API_KEY: API key for the LangSmith tracing service.
    LANGSMITH_PROJECT: LangSmith project name traces are grouped under.

Typical usage::

    from auto_parts_research_agent.workflow import run_research

    result = run_research("2018 Honda Civic vibration at highway speeds")
    print(result["recommendation"])
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import argparse   # Parses command-line arguments when the module is run as a script.
import json       # Encodes/decodes JSON payloads returned by the LLM planner.
import os         # Reads environment variables (API keys, model names, URLs).
import re         # Regular-expression helpers for stripping <think> blocks and
                  # extracting JSON objects from LLM output.
from typing import Literal, TypedDict  # Static typing helpers used throughout.

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
# load_dotenv() reads key=value pairs from a ``.env`` file in the project root
# and injects them into the process environment before any other code runs.
# This keeps secrets (API keys, URLs) out of source control.

from langchain_community.tools import DuckDuckGoSearchRun
# DuckDuckGoSearchRun wraps the DuckDuckGo instant-answer API.  It requires no
# API key and returns a plain-text search-result snippet for a given query string.

from langchain_core.messages import HumanMessage, SystemMessage
# LangChain message objects used to construct structured chat prompts:
#   SystemMessage  – sets the persona / instructions for the LLM.
#   HumanMessage   – represents the user turn that contains the actual request.

from langchain_openai import ChatOpenAI
# ChatOpenAI is a LangChain wrapper around any OpenAI-compatible chat endpoint.
# By overriding ``base_url`` and ``api_key`` we can point it at a local Ollama
# server instead of the real OpenAI cloud API.

from langgraph.graph import END, START, StateGraph
# StateGraph  – the directed-graph builder; nodes are Python functions, edges
#               define execution order.
# START / END – sentinel constants that mark the entry and exit of the graph.

from langsmith import traceable
# The @traceable decorator wraps any Python function so that its inputs, outputs,
# and latency are automatically recorded as a "run" in the LangSmith tracing UI.
# Tracing is only active when LANGSMITH_TRACING=true is set in the environment.

# ---------------------------------------------------------------------------
# Load environment variables from .env before any env reads happen below.
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Shared workflow state
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict):
    """Mutable state dictionary passed between every node in the LangGraph graph.

    LangGraph nodes receive a copy of this dict, modify it, and return it.
    The graph runtime merges the returned values back into the shared state
    before calling the next node.

    Attributes:
        vehicle_problem (str): The raw, plain-English problem description
            supplied by the operator (e.g. "2014 Ford Focus rough idle").
            This value is set once at the start and never mutated.

        plan (list[str]): Ordered list of research steps produced by the
            planner agent.  Each entry is a concise directive such as
            "Identify likely failure components".

        current_step (int): Zero-based index into ``plan`` that tracks which
            step the research agent should execute next.  Incremented by one
            after each successful research iteration.

        findings (list[str]): Human-readable progress log appended to by the
            research agent after each step completes.  Useful for debugging
            and audit trails.

        raw_search_notes (list[str]): Full, unprocessed text blocks returned
            by DuckDuckGo for each research step.  Passed verbatim to the
            summarizer so the LLM can synthesise them.

        draft_summary (str): The structured recommendation text produced by
            the summarizer agent.  Presented to the human for approval.

        approved (bool): Set to ``True`` by the human checkpoint agent when
            the operator types "Y" or "yes".  Controls graph routing after
            the checkpoint.

        operator_feedback (str): Optional revision instructions entered by the
            operator when they reject a draft.  Passed to the summarizer on
            the next iteration so the LLM can adjust tone, depth, or focus.
            Reset to an empty string each time the operator approves or starts
            a fresh rejection cycle.

        recommendation (str): The final, operator-approved recommendation.
            Populated by the finalize agent only when ``approved`` is ``True``.
            Empty string if the run ends without approval.
    """

    vehicle_problem: str
    plan: list[str]
    current_step: int
    findings: list[str]
    raw_search_notes: list[str]
    draft_summary: str
    approved: bool
    operator_feedback: str
    recommendation: str


# ---------------------------------------------------------------------------
# Helper: default state factory
# ---------------------------------------------------------------------------

def _default_state(vehicle_problem: str) -> WorkflowState:
    """Return a freshly initialised :class:`WorkflowState` for a new run.

    All list fields start empty and scalar fields are set to safe zero/empty
    values so every node can safely read them without checking for ``None``.

    Args:
        vehicle_problem: Plain-English description of the vehicle fault the
            operator wants researched (e.g. "2018 Honda Civic brake squeal").

    Returns:
        A fully populated :class:`WorkflowState` ready to be passed to
        ``app.invoke()``.
    """
    return {
        "vehicle_problem": vehicle_problem,
        "plan": [],            # Filled by planner_agent.
        "current_step": 0,     # Incremented by research_agent each iteration.
        "findings": [],        # Audit log appended to by research_agent.
        "raw_search_notes": [], # Raw DuckDuckGo snippets collected per step.
        "draft_summary": "",      # Written by summarizer_agent.
        "approved": False,        # Flipped to True by human_checkpoint_agent.
        "operator_feedback": "",  # Revision notes captured on rejection.
        "recommendation": "",     # Written by finalize_agent on approval.
    }


# ---------------------------------------------------------------------------
# Helper: strip <think> … </think> blocks
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove extended-thinking markup emitted by reasoning-capable models.

    Some local models (e.g. Qwen3, DeepSeek-R1) prefix their actual answer
    with a ``<think>`` block that contains raw chain-of-thought reasoning.
    This internal monologue is useful for the model but breaks downstream
    JSON parsing and clutters the final recommendation shown to the operator.

    The regex uses ``re.DOTALL`` so that ``.`` matches newline characters,
    allowing multi-line thinking blocks to be removed in a single pass.

    Args:
        text: Raw string returned by ``response.content`` from the LLM.

    Returns:
        The input string with all ``<think>...</think>`` sections removed
        and leading/trailing whitespace stripped.

    Example::

        >>> raw = "<think>Let me reason...</think>\\n{\"plan\": [\"step1\"]}"
        >>> _strip_thinking(raw)
        '{"plan": ["step1"]}'
    """
    # re.DOTALL makes '.' match '\n' so multi-line <think> blocks are caught.
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Helper: build the LLM client
# ---------------------------------------------------------------------------

def _maybe_llm() -> ChatOpenAI:
    """Construct and return a :class:`~langchain_openai.ChatOpenAI` instance.

    All configuration is read from environment variables (populated from
    ``.env`` by :func:`load_dotenv` at module import time):

    - ``OPENAI_API_KEY``  – Required by the OpenAI client library but can be
      any non-empty string when talking to Ollama (defaults to ``"ollama"``).
    - ``OPENAI_BASE_URL`` – The root URL of the OpenAI-compatible API.
      Defaults to ``http://localhost:11434/v1`` (local Ollama).
    - ``OPENAI_MODEL``    – Model tag understood by the endpoint.
      Defaults to ``qwen3.5:latest``.

    ``temperature=0`` is set to make outputs deterministic and consistent
    across repeated runs — important for structured JSON responses from the
    planner and for reproducible recommendations.

    Returns:
        A configured :class:`~langchain_openai.ChatOpenAI` client ready to
        call ``.invoke()`` with a list of LangChain message objects.
    """
    return ChatOpenAI(
        # The model tag sent in every API request body (e.g. "qwen3.5:latest").
        model=os.getenv("OPENAI_MODEL", "qwen3.5:latest"),

        # temperature=0 disables sampling randomness: the model always picks
        # the highest-probability token, giving deterministic output.
        temperature=0,

        # Ollama doesn't validate the API key but the OpenAI client requires
        # a non-empty string — "ollama" is a conventional placeholder.
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),

        # Redirect all requests to the local Ollama server instead of
        # api.openai.com.  The /v1 suffix matches the OpenAI REST spec.
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    )


# ---------------------------------------------------------------------------
# Node 1 — Planner agent
# ---------------------------------------------------------------------------

@traceable(name="planner_agent")
def planner_agent(state: WorkflowState) -> WorkflowState:
    """Decompose the vehicle problem into an ordered list of research steps.

    This is the first node in the LangGraph graph.  It sends the operator's
    raw problem description to the LLM with a strict system prompt that
    instructs the model to return **only** a JSON object of the form::

        {"plan": ["step 1", "step 2", "step 3"]}

    The planner uses two layers of JSON extraction to handle common LLM
    quirks:

    1. ``_strip_thinking()`` removes any ``<think>`` block prepended by
       reasoning models before the actual answer.
    2. A regex ``re.search(r"\\{.*\\}", …)`` extracts the first JSON object
       from the response in case the model wraps it in Markdown fences
       (e.g. ```json … ```).
    3. If both JSON parses fail, the response is split into lines and the
       first four non-empty lines are used as plan steps (best-effort fallback).

    The completed plan is printed to stdout for operator visibility, then
    stored in ``state["plan"]``.  ``state["current_step"]`` is reset to 0 so
    the research loop always starts from the first step.

    Args:
        state: The current :class:`WorkflowState`.  Only ``state["vehicle_problem"]``
            is read; ``state["plan"]`` and ``state["current_step"]`` are written.

    Returns:
        The updated :class:`WorkflowState` with ``plan`` and ``current_step`` set.
    """
    llm = _maybe_llm()
    problem = state["vehicle_problem"]

    print("[planner] Building research plan...")

    # System prompt: instructs the model to act as a diagnostics planner and
    # enforces a strict JSON-only output contract so we can parse it reliably.
    prompt = (
        "You are an automotive diagnostics planner for an auto-parts research workflow. "
        "Return ONLY a JSON object with key 'plan' as an array of 3-5 concise steps. "
        "Do not include any explanation or markdown — only valid JSON. "
        "Focus on diagnosis + parts recommendation."
    )

    # Send the two-turn prompt (system persona + user problem) to the LLM.
    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=problem)])

    # Strip any <think>...</think> reasoning block the model may have emitted
    # before the actual JSON answer.
    content = _strip_thinking(str(response.content))

    try:
        # Attempt 1: find the first {...} block in the response using a greedy
        # regex.  re.DOTALL lets '.' span newlines so multi-line JSON is caught.
        match = re.search(r"\{.*\}", content, re.DOTALL)

        # If a JSON-shaped block was found, parse it; otherwise try the full
        # stripped content directly (handles the case where the model returned
        # clean JSON without any surrounding text).
        payload = json.loads(match.group() if match else content)
        plan = payload["plan"]

    except Exception:
        # Attempt 2 (fallback): the model returned something we couldn't parse
        # as JSON at all.  Split the text into lines, strip list markers
        # ("- ", "* ", etc.) and take the first four non-empty lines as steps.
        plan = [line.strip("- ") for line in content.splitlines() if line.strip()][:4]

    # Print each plan step so the operator can see the research agenda.
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    # Persist the plan and reset the step counter for the research loop.
    state["plan"] = plan
    state["current_step"] = 0
    return state


# ---------------------------------------------------------------------------
# Node 2 — Research agent
# ---------------------------------------------------------------------------

@traceable(name="research_agent")
def research_agent(state: WorkflowState) -> WorkflowState:
    """Execute a single DuckDuckGo search for the current plan step.

    This node is called **repeatedly** by the LangGraph graph — once per plan
    step — via the conditional edge :func:`should_continue_research`.  Each
    invocation handles exactly one step (identified by ``state["current_step"]``)
    and increments the counter before returning, so the next invocation picks
    up the following step automatically.

    The search query is constructed as a structured string that includes both
    the overall vehicle problem and the specific step directive.  This gives
    DuckDuckGo enough context to return relevant technical results rather than
    generic automotive content.

    If DuckDuckGo raises any exception (rate limit, network error, etc.) the
    error message is stored as the research note so the run continues rather
    than crashing.  The summarizer will include the error text in its evidence
    and can note the gap in its recommendation.

    Args:
        state: The current :class:`WorkflowState`.  Reads ``current_step``,
            ``plan``, and ``vehicle_problem``; appends to ``raw_search_notes``
            and ``findings``; increments ``current_step``.

    Returns:
        The updated :class:`WorkflowState` with new entries in
        ``raw_search_notes`` and ``findings`` and an incremented
        ``current_step``.
    """
    # Capture the index before we mutate it so all references in this call
    # consistently refer to the same step.
    step_index = state["current_step"]
    plan = state["plan"]

    # Safety guard: if the counter somehow exceeds the plan length (e.g. due to
    # a graph wiring bug) return early without doing any work.
    if step_index >= len(plan):
        return state

    step = plan[step_index]

    # Initialise DuckDuckGo search tool.  No API key required; results are
    # returned as a single plain-text snippet (not a list of URLs).
    search = DuckDuckGoSearchRun()

    # Build a query that gives DuckDuckGo two pieces of context:
    #   1. The full vehicle problem (make/model/year/symptom).
    #   2. The specific research directive for this step.
    query = f"vehicle issue: {state['vehicle_problem']} | research step: {step}"

    try:
        results = search.run(query)
    except Exception as exc:
        # Capture the exception message so the run can continue.  The
        # summarizer will see this and can caveat the recommendation accordingly.
        results = f"Search unavailable: {exc}"

    # Log progress to stdout so the operator can track which step is running.
    print(f"[research] Step {step_index + 1}/{len(plan)}: {step}")

    # Build the raw note: a labelled block containing the step description and
    # the full DuckDuckGo snippet.  Stored verbatim for the summarizer.
    note = f"Step {step_index + 1}: {step}\nResearch Notes: {results}"
    state["raw_search_notes"].append(note)

    # Append a short human-readable completion entry to the findings audit log.
    state["findings"].append(f"Completed research step {step_index + 1}: {step}")

    # Advance the step counter so the next invocation of this node (if any)
    # processes the following plan step.
    state["current_step"] += 1
    return state


# ---------------------------------------------------------------------------
# Conditional edge — should research continue?
# ---------------------------------------------------------------------------

def should_continue_research(state: WorkflowState) -> Literal["research", "summarize"]:
    """Decide whether to run another research iteration or move to summarisation.

    This function is registered as a conditional edge in the LangGraph graph.
    After every execution of :func:`research_agent`, LangGraph calls this
    function to determine which node to visit next.

    The logic is simple: if the step counter hasn't reached the end of the
    plan list, route back to ``research`` so the next step is processed.
    Once all steps are done, route forward to ``summarize``.

    Args:
        state: The current :class:`WorkflowState`.  Only ``current_step`` and
            ``plan`` are inspected.

    Returns:
        ``"research"`` if there are unprocessed plan steps remaining, or
        ``"summarize"`` when all steps have been completed.
    """
    # current_step is incremented *after* each research call, so when it equals
    # len(plan) all steps have been processed and we can move on.
    if state["current_step"] < len(state["plan"]):
        return "research"
    return "summarize"


# ---------------------------------------------------------------------------
# Node 3 — Summarizer agent
# ---------------------------------------------------------------------------

@traceable(name="summarizer_agent")
def summarizer_agent(state: WorkflowState) -> WorkflowState:
    """Synthesise all research notes into a structured recommendation draft.

    This node sends the full body of evidence collected by the research agent
    to the LLM together with a system prompt that instructs it to produce a
    four-section recommendation:

    - **Likely Cause**       – Most probable root cause based on the evidence.
    - **Validation Steps**   – How to confirm the diagnosis before ordering parts.
    - **Suggested Part Categories** – OEM/aftermarket part types to investigate.
    - **Risk / Confidence**  – How certain the recommendation is and what gaps remain.

    The ``<think>`` stripping step is applied to the LLM response here as well,
    since reasoning models may prepend internal monologue before the actual
    recommendation text.

    Args:
        state: The current :class:`WorkflowState`.  Reads ``raw_search_notes``
            and ``vehicle_problem``; writes ``draft_summary``.

    Returns:
        The updated :class:`WorkflowState` with ``draft_summary`` populated.
    """
    llm = _maybe_llm()
    print("[summarizer] Generating recommendation draft...")

    # System prompt defining the output structure the LLM must follow.
    prompt = (
        "You are a senior parts advisor. Build a concise recommendation with sections: "
        "Likely Cause, Validation Steps, Suggested Part Categories, and Risk/Confidence."
    )

    # Concatenate all raw research notes into a single evidence block.
    # Each note is already labelled with its step number and directive, so the
    # LLM has full traceability of where each piece of evidence came from.
    evidence = "\n\n".join(state["raw_search_notes"])

    # If the operator rejected a previous draft and left revision notes, append
    # them to the user turn so the LLM knows exactly what to change.  The
    # section is only added when feedback is non-empty to keep the prompt clean
    # on the first summarisation pass.
    feedback_section = ""
    if state.get("operator_feedback"):
        feedback_section = f"\n\nOperator revision notes:\n{state['operator_feedback']}"

    # Two-turn prompt: system persona sets the advisory role, user turn provides
    # the problem statement, all collected evidence, and any operator feedback.
    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(
                content=(
                    f"Problem:\n{state['vehicle_problem']}\n\n"
                    f"Evidence:\n{evidence}"
                    f"{feedback_section}"
                )
            ),
        ]
    )

    # Strip any <think> block before storing the summary — we only want the
    # clean, human-readable recommendation text.
    summary = _strip_thinking(str(response.content))

    state["draft_summary"] = summary
    return state


# ---------------------------------------------------------------------------
# Node 4 — Formatter agent
# ---------------------------------------------------------------------------

@traceable(name="formatter_agent")
def formatter_agent(state: WorkflowState) -> WorkflowState:
    """Rewrite the technical draft summary into a clean, plain-English report.

    The summarizer agent produces a content-rich but often dense recommendation.
    This node passes that draft through the LLM a second time with a different
    persona — a friendly service advisor explaining findings to a car owner —
    so the final text presented at the human checkpoint is approachable and
    easy to act on, regardless of the reader's technical background.

    The formatter does **not** change the factual content or add new information;
    it only improves presentation.  The rewritten text overwrites ``draft_summary``
    in place so the human checkpoint and finalize nodes see the polished version
    without any additional state fields.

    Formatting guidelines applied by the LLM:
    - Clear, numbered or bulleted sections with bold headings.
    - Short sentences and plain language — no unexplained jargon.
    - Actionable language (e.g. "Take it to a shop to…" rather than "Validation
      should be performed…").
    - A brief plain-English summary paragraph at the top before the sections,
      so the reader immediately understands the bottom line.

    Args:
        state: The current :class:`WorkflowState`.  Reads ``draft_summary``;
            overwrites ``draft_summary`` with the formatted version.

    Returns:
        The updated :class:`WorkflowState` with ``draft_summary`` replaced by
        the human-friendly formatted report.
    """
    llm = _maybe_llm()
    print("[formatter] Polishing summary for readability...")

    # System prompt: shifts the LLM's persona from technical analyst to
    # friendly service advisor writing for a non-technical car owner.
    prompt = (
        "You are a friendly automotive service advisor writing a report for a car owner "
        "who is not a mechanic. Rewrite the following technical recommendation so it is "
        "easy to understand and pleasant to read. Follow these rules:\n"
        "1. Start with a short 2-3 sentence plain-English summary of the bottom line.\n"
        "2. Use bold headings for each section.\n"
        "3. Use bullet points for lists of steps or parts.\n"
        "4. Replace technical jargon with simple language, or explain it in parentheses.\n"
        "5. Use active, actionable language (e.g. 'Check the spark plugs' not "
        "'Spark plug inspection should be performed').\n"
        "6. Keep the same four sections: Likely Cause, Validation Steps, "
        "Suggested Parts, and Confidence Level.\n"
        "Do not add new facts or remove any important information from the original."
    )

    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            # Pass the raw technical draft as the content to be reformatted.
            HumanMessage(content=state["draft_summary"]),
        ]
    )

    # Strip any <think> block and overwrite the draft with the polished version.
    state["draft_summary"] = _strip_thinking(str(response.content))
    return state


# ---------------------------------------------------------------------------
# Node 5 — Human-in-the-loop checkpoint
# ---------------------------------------------------------------------------

@traceable(name="human_checkpoint_agent")
def human_checkpoint_agent(state: WorkflowState) -> WorkflowState:
    """Present the recommendation draft to the operator and capture approval.

    This node implements the human-in-the-loop (HITL) pattern: the workflow
    pauses execution and blocks on ``input()`` until the operator explicitly
    types "y" or "yes" to approve, or any other input to reject.

    If the operator rejects the draft, the conditional edge
    :func:`route_after_checkpoint` routes the graph back to
    :func:`summarizer_agent`, which regenerates the draft.  This loop
    continues until the operator approves or terminates the process.

    .. note::
        Because this node calls ``input()``, it is **not** suitable for
        automated or headless deployments.  For non-interactive use cases,
        replace this node with a pre-approval stub that always sets
        ``state["approved"] = True``.

    Args:
        state: The current :class:`WorkflowState`.  Reads ``draft_summary``;
            writes ``approved``.

    Returns:
        The updated :class:`WorkflowState` with ``approved`` set to ``True``
        if the operator approved, ``False`` otherwise.
    """
    # Display the full draft recommendation between clear dividers so it is
    # easy to read in a terminal environment.
    print("\n========== HUMAN CHECKPOINT ==========")
    print(state["draft_summary"])
    print("=====================================\n")

    # Block and wait for the operator's decision.  strip() removes accidental
    # leading/trailing spaces; lower() makes the check case-insensitive.
    # [Y/n] — uppercase Y signals that approval is the expected/default choice.
    answer = input("Approve this recommendation draft? [Y/n]: ").strip().lower()

    # Accept "y", "yes", or a bare Enter (empty string) as affirmative answers.
    # Any other input (e.g. "n", "no", "redo") is treated as rejection.
    approved = answer in {"y", "yes", ""}
    state["approved"] = approved

    if not approved:
        # Prompt the operator for specific revision instructions.  These notes
        # are stored in state and injected into the next summariser call so the
        # LLM knows exactly what to change (e.g. "focus more on electrical
        # components" or "add OEM part numbers").
        feedback = input("What should be changed? Provide revision notes for the AI: ").strip()
        state["operator_feedback"] = feedback

    return state


# ---------------------------------------------------------------------------
# Conditional edge — route after human checkpoint
# ---------------------------------------------------------------------------

def route_after_checkpoint(state: WorkflowState) -> Literal["finalize", "summarize"]:
    """Route to finalisation on approval, or back to summarisation on rejection.

    Called by LangGraph as a conditional edge after :func:`human_checkpoint_agent`
    completes.  Reads the ``approved`` flag set by the checkpoint node and
    returns the name of the next node to execute.

    Args:
        state: The current :class:`WorkflowState`.  Only ``approved`` is read.

    Returns:
        ``"finalize"`` if the operator approved the draft, or ``"summarize"``
        to regenerate the recommendation and present it again.
    """
    return "finalize" if state["approved"] else "summarize"


# ---------------------------------------------------------------------------
# Node 6 — Finalize agent
# ---------------------------------------------------------------------------

@traceable(name="finalize_agent")
def finalize_agent(state: WorkflowState) -> WorkflowState:
    """Promote the approved draft summary to the official recommendation field.

    This is the last node in the graph before the ``END`` sentinel.  Its sole
    responsibility is to copy ``draft_summary`` into ``recommendation`` when
    the operator has given approval.

    The explicit ``approved`` guard is a safety check: even though the graph
    routing logic should prevent this node from running without approval, the
    guard ensures ``recommendation`` is never populated from an unapproved draft
    if the graph is rewired or called programmatically.

    Args:
        state: The current :class:`WorkflowState`.  Reads ``approved`` and
            ``draft_summary``; conditionally writes ``recommendation``.

    Returns:
        The updated :class:`WorkflowState`.  If approved, ``recommendation``
        contains the final text.  If not approved (defensive path),
        ``recommendation`` remains an empty string.
    """
    if state["approved"]:
        # Copy the operator-approved draft into the canonical output field.
        state["recommendation"] = state["draft_summary"]
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Assemble and compile the LangGraph StateGraph for the research pipeline.

    Nodes are Python functions that accept and return a :class:`WorkflowState`
    dict.  Edges are either unconditional (always go to the next node) or
    conditional (a routing function decides the next node at runtime).

    Graph topology::

        START
          │
          ▼
        planner ──────────────────► research ◄──────────────────┐
                                        │                        │
                              should_continue_research           │
                                   │         │                   │
                             "research"  "summarize"             │
                                   │         │                   │
                                   └─────────┤                   │
                                             ▼                   │
                                         summarize               │
                                             │                   │
                                             ▼                   │
                                         formatter               │
                                             │                   │
                                             ▼                   │
                                    human_checkpoint             │
                                             │                   │
                               route_after_checkpoint            │
                                   │              │              │
                              "finalize"    "summarize" ─────────┘
                                   │
                                   ▼
                                 END

    Returns:
        A compiled LangGraph application (``CompiledGraph``) that can be
        invoked with ``app.invoke(state_dict)``.
    """
    graph = StateGraph(WorkflowState)

    # Register each agent function as a named node in the graph.
    graph.add_node("planner", planner_agent)
    graph.add_node("research", research_agent)
    graph.add_node("summarize", summarizer_agent)
    graph.add_node("formatter", formatter_agent)
    graph.add_node("human_checkpoint", human_checkpoint_agent)
    graph.add_node("finalize", finalize_agent)

    # Unconditional edge: the run always starts with the planner.
    graph.add_edge(START, "planner")

    # Unconditional edge: after planning, always enter the research loop.
    graph.add_edge("planner", "research")

    # Conditional edge: after each research step, decide whether to loop back
    # to research (more steps remain) or advance to summarisation (all done).
    graph.add_conditional_edges(
        "research",
        should_continue_research,
        {"research": "research", "summarize": "summarize"},
    )

    # Unconditional edge: after summarisation, always polish for readability.
    graph.add_edge("summarize", "formatter")

    # Unconditional edge: after formatting, present the polished draft for review.
    graph.add_edge("formatter", "human_checkpoint")

    # Conditional edge: after human review, either finalise (approved) or
    # regenerate the summary (rejected) and loop back through human_checkpoint.
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_checkpoint,
        {"finalize": "finalize", "summarize": "summarize"},
    )

    # Unconditional edge: finalisation is always the last step before END.
    graph.add_edge("finalize", END)

    # compile() validates the graph (checks for unreachable nodes, missing
    # edges, etc.) and returns an executable application object.
    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@traceable(name="run_research")
def run_research(vehicle_problem: str) -> WorkflowState:
    """Run the complete research pipeline for a given vehicle problem.

    Builds a fresh LangGraph application, initialises the workflow state with
    the provided problem description, and executes the full agent pipeline
    synchronously, blocking until the operator approves a recommendation or
    the process is terminated.

    This is the primary public API of the module.  Callers that want to
    integrate the agent into a larger system should call this function rather
    than constructing the graph manually.

    Args:
        vehicle_problem: A plain-English description of the vehicle fault to
            research.  Should include the vehicle year, make, model, and a
            clear symptom description for best results.
            Example: ``"2014 Ford Focus rough idle and intermittent CEL"``.

    Returns:
        The final :class:`WorkflowState` after all nodes have executed.
        The ``recommendation`` key contains the operator-approved text, or
        an empty string if the run ended without approval.
    """
    app = build_graph()
    # invoke() runs the graph synchronously and returns the final state dict.
    return app.invoke(_default_state(vehicle_problem))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for the Auto Parts Research Agent.

    Parses a single optional positional argument (the vehicle problem string)
    and runs the full research pipeline.  If no argument is provided, a default
    example problem is used so the tool can be tested without any arguments.

    The final approved recommendation (or a "no recommendation" notice if the
    operator never approved a draft) is printed to stdout after the pipeline
    completes.

    CLI usage::

        # Use the default example problem:
        python -m auto_parts_research_agent.cli

        # Supply a custom problem:
        python -m auto_parts_research_agent.cli "2020 Toyota Camry AC not cooling"
    """
    parser = argparse.ArgumentParser(description="Run the Auto Parts Research Agent workflow.")
    parser.add_argument(
        "problem",
        nargs="?",  # Makes the argument optional; falls back to ``default`` if omitted.
        default="2014 Ford Focus rough idle and intermittent check engine light",
        help="Vehicle problem description to research.",
    )
    args = parser.parse_args()

    result = run_research(args.problem)

    print("\n===== FINAL RECOMMENDATION =====")
    # ``recommendation`` is an empty string when the operator never approved a
    # draft, so we display a clear notice rather than printing a blank line.
    print(result.get("recommendation") or "No recommendation approved.")


# ---------------------------------------------------------------------------
# Script guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allows the module to be executed directly with:
    #   python src/auto_parts_research_agent/workflow.py
    # In normal usage the ``auto-parts-agent`` CLI entry point (defined in
    # pyproject.toml) calls main() via auto_parts_research_agent.cli instead.
    main()
