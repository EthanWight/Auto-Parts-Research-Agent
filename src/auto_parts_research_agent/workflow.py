"""Core LangGraph multi-agent workflow for the Auto Parts Research Agent.

This module defines the agentic pipeline that accepts a vehicle problem
description and produces a structured parts/repair recommendation. The
pipeline is a directed graph (LangGraph) with six agent nodes:

1. The planner agent decomposes the problem into 3-5 research steps via the LLM.
2. The research agent runs a DuckDuckGo search for each step and stores raw notes.
3. The summarizer agent synthesizes research notes into a recommendation draft.
4. The formatter agent rewrites the draft in plain, human-friendly language.
5. The human checkpoint agent presents the draft and waits for operator approval.
6. The finalize agent promotes the approved draft to the final recommendation.

Environment variables (loaded from ".env"):
    OPENAI_API_KEY: Set to "ollama" when using a local Ollama server.
    OPENAI_BASE_URL: OpenAI-compatible API endpoint (default: http://localhost:11434/v1).
    OPENAI_MODEL: Model identifier for the endpoint (default: qwen3.5:latest).
    LANGSMITH_TRACING: Set to "true" to send traces to LangSmith.
    LANGSMITH_API_KEY: API key for LangSmith.
    LANGSMITH_PROJECT: LangSmith project name.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

# Load environment variables from .env before any os.getenv() calls.
load_dotenv()


class WorkflowState(TypedDict):
    """State dictionary shared across all nodes in the LangGraph pipeline.

    Each node receives this dict, modifies relevant fields, and returns it.
    The graph runtime merges updates back into the shared state automatically.

    Args:
        vehicle_problem: Plain-English problem description from the operator.
        plan: Ordered research steps produced by the planner.
        current_step: Zero-based index tracking the next research step.
        findings: Human-readable audit log of completed research steps.
        raw_search_notes: Unprocessed DuckDuckGo result blocks per step.
        draft_summary: Recommendation text from the summarizer/formatter.
        approved: True when the operator approves the draft.
        operator_feedback: Revision notes entered on rejection, passed to the
            summarizer on the next iteration.
        recommendation: Final operator-approved recommendation text.
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


def _default_state(vehicle_problem: str) -> WorkflowState:
    """Create a fresh WorkflowState with safe defaults for a new run.

    Args:
        vehicle_problem: Description of the vehicle fault to research.

    Returns:
        A fully initialized WorkflowState dictionary.
    """
    return {
        "vehicle_problem": vehicle_problem,
        "plan": [],
        "current_step": 0,
        "findings": [],
        "raw_search_notes": [],
        "draft_summary": "",
        "approved": False,
        "operator_feedback": "",
        "recommendation": "",
    }


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models.

    Models like Qwen3 and DeepSeek-R1 prepend chain-of-thought reasoning
    inside <think> tags. This helper strips those blocks, so downstream
    parsing (JSON extraction, display) receives only the real answer.

    Args:
        text: Raw LLM response string.

    Returns:
        The input with all <think>...</think> sections is removed and
        whitespace trimmed.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _maybe_llm() -> ChatOpenAI:
    """Build and return a ChatOpenAI client pointing at the configured endpoint.

    Reads OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL from the
    environment. Defaults target a local Ollama server with the
    qwen3.5:latest model. Temperature is pinned to 0 for deterministic output.

    Returns:
        A configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "qwen3.5:latest"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),  # type: ignore[arg-type]
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    )


# ---------------------------------------------------------------------------
# Node 1 - Planner
# ---------------------------------------------------------------------------

@traceable(name="planner_agent")
def planner_agent(state: WorkflowState) -> WorkflowState:
    """Decompose the vehicle problem into 3-5 ordered research steps.

    This function sends the problem to the LLM with a strict "JSON only"
    system prompt. It applies two fallback layers to handle quirky LLM output:
    1. Strips any <think> block, then extracts the first JSON object via regex.
    2. If JSON parsing fails, splits the response into lines and takes the
       first four non-empty lines as plan steps.

    Args:
        state: Current workflow state. Reads "vehicle_problem", writes "plan"
            and resets "current_step" to 0.

    Returns:
        The updated state with the research plan populated.
    """
    llm = _maybe_llm()
    problem = state["vehicle_problem"]

    print("[planner] Building research plan...")

    prompt = (
        "You are an automotive diagnostics planner for an auto-parts research workflow. "
        "Return ONLY a JSON object with key 'plan' as an array of 3-5 concise steps. "
        "Do not include any explanation or markdown - only valid JSON. "
        "Focus on diagnosis + parts recommendation."
    )

    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=problem)])
    content = _strip_thinking(str(response.content))

    try:
        # Extract the first JSON object from the response.
        match = re.search(r"\{.*}", content, re.DOTALL)
        payload = json.loads(match.group() if match else content)
        plan = payload["plan"]
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        # Fallback: treat each non-empty line as a plan step.
        plan = [line.strip("- ") for line in content.splitlines() if line.strip()][:4]

    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")

    state["plan"] = plan
    state["current_step"] = 0
    return state


# ---------------------------------------------------------------------------
# Node 2 - Research
# ---------------------------------------------------------------------------

@traceable(name="research_agent")
def research_agent(state: WorkflowState) -> WorkflowState:
    """Run a DuckDuckGo search for the current plan step.

    Called repeatedly by the graph (once per plan step) via the
    "should_continue_research" conditional edge. Each call handles one step,
    increments "current_step", and appends raw results to "raw_search_notes".

    If the search raises an exception, the error message is stored instead,
    so the pipeline continues without crashing.

    Args:
        state: Current workflow state. Reads "current_step", "plan", and
            "vehicle_problem". Appends to "raw_search_notes" and "findings",
            then increments "current_step".

    Returns:
        The updated state with new research notes recorded.
    """
    step_index = state["current_step"]
    plan = state["plan"]

    if step_index >= len(plan):
        return state

    step = plan[step_index]
    search = DuckDuckGoSearchRun()
    query = f"vehicle issue: {state['vehicle_problem']} | research step: {step}"

    try:
        results = search.run(query)
    except Exception as exc:
        results = f"Search unavailable: {exc}"

    print(f"[research] Step {step_index + 1}/{len(plan)}: {step}")

    note = f"Step {step_index + 1}: {step}\nResearch Notes: {results}"
    state["raw_search_notes"].append(note)
    state["findings"].append(f"Completed research step {step_index + 1}: {step}")
    state["current_step"] += 1
    return state


# ---------------------------------------------------------------------------
# Conditional edge - research loop
# ---------------------------------------------------------------------------

def should_continue_research(state: WorkflowState) -> Literal["research", "summarize"]:
    """Route back to research if steps remain, or move to summarization.

    Args:
        state: Current workflow state. Inspects "current_step" and "plan".

    Returns:
        The string "research" if unprocessed steps remain, or "summarize"
        when all steps are done.
    """
    if state["current_step"] < len(state["plan"]):
        return "research"
    return "summarize"


# ---------------------------------------------------------------------------
# Node 3 - Summarizer
# ---------------------------------------------------------------------------

@traceable(name="summarizer_agent")
def summarizer_agent(state: WorkflowState) -> WorkflowState:
    """Synthesize all research notes into a structured recommendation draft.

    Sends the collected evidence to the LLM with a prompt requesting four
    sections: Likely Cause, Validation Steps, Suggested Part Categories,
    and Risk/Confidence. If the operator previously rejected a draft and
    left feedback, those revision notes are appended to the prompt.

    Args:
        state: Current workflow state. Reads "raw_search_notes",
            "vehicle_problem", and "operator_feedback". Writes "draft_summary".

    Returns:
        The updated state with the draft summary populated.
    """
    llm = _maybe_llm()
    print("[summarizer] Generating recommendation draft...")

    prompt = (
        "You are a senior parts advisor. Build a concise recommendation with sections: "
        "Likely Cause, Validation Steps, Suggested Part Categories, and Risk/Confidence."
    )

    evidence = "\n\n".join(state["raw_search_notes"])

    # Append operator revision notes when present (empty on first pass).
    feedback_section = ""
    if state.get("operator_feedback"):
        feedback_section = f"\n\nOperator revision notes:\n{state['operator_feedback']}"

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

    state["draft_summary"] = _strip_thinking(str(response.content))
    return state


# ---------------------------------------------------------------------------
# Node 4 - Formatter
# ---------------------------------------------------------------------------

@traceable(name="formatter_agent")
def formatter_agent(state: WorkflowState) -> WorkflowState:
    """Rewrite the technical draft into a clean, plain-English report.

    Passes the summarizer's output through the LLM a second time with a
    "friendly service advisor" persona. The formatter keeps all facts intact
    but rewrites them with bullet points, bold headings, short sentences,
    and jargon-free language so a non-mechanic car owner can understand.

    Args:
        state: Current workflow state. Reads "draft_summary" and overwrites
            it with the formatted version.

    Returns:
        The updated state with a human-friendly draft summary.
    """
    llm = _maybe_llm()
    print("[formatter] Polishing summary for readability...")

    prompt = (
        "You are a friendly automotive service advisor writing a report for a car owner "
        "who is not a mechanic. Rewrite the following technical recommendation so it is "
        "easy to understand and pleasant to read. Follow these rules:\n"
        "1. Start with a short 2-3 sentence plain-English summary of the bottom line.\n"
        "2. Use bold headings for each section.\n"
        "3. Use bullet points for lists of steps or parts.\n"
        "4. Replace technical jargon with simple language, or explain it in parentheses.\n"
        "5. Use active, actionable language ('Check the spark plugs' not "
        "'Spark plug inspection should be performed').\n"
        "6. Keep the same four sections: Likely Cause, Validation Steps, "
        "Suggested Parts, and Confidence Level.\n"
        "Do not add new facts or remove any important information from the original."
    )

    response = llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=state["draft_summary"]),
        ]
    )

    state["draft_summary"] = _strip_thinking(str(response.content))
    return state


# ---------------------------------------------------------------------------
# Node 5 - Human checkpoint
# ---------------------------------------------------------------------------

@traceable(name="human_checkpoint_agent")
def human_checkpoint_agent(state: WorkflowState) -> WorkflowState:
    """Present the draft to the operator and capture approval or feedback.

    Prints the formatted recommendation and blocks on input(). The operator
    types Y/yes/Enter to approve, or anything else to reject. On rejection,
    the operator is prompted for revision notes that get fed back into
    the summarizer on the next pass.

    Note: This node calls input() and is not suitable for headless runs.

    Args:
        state: Current workflow state. Reads "draft_summary", writes
            "approved" and (on rejection) "operator_feedback".

    Returns:
        The updated state with the approval decision is recorded.
    """
    print("\n========== HUMAN CHECKPOINT ==========")
    print(state["draft_summary"])
    print("=====================================\n")

    answer = input("Approve this recommendation draft? [Y/N]: ").strip().lower()

    # Y, yes, or bare Enter all count as approval.
    approved = answer in {"y", "yes", "", "Y"}
    state["approved"] = approved

    if not approved:
        feedback = input("What should be changed? Provide revision notes for the AI: ").strip()
        state["operator_feedback"] = feedback

    return state


# ---------------------------------------------------------------------------
# Conditional edge - post-checkpoint routing
# ---------------------------------------------------------------------------

def route_after_checkpoint(state: WorkflowState) -> Literal["finalize", "summarize"]:
    """Route to finalization on approval, or back to summarization on rejection.

    Args:
        state: Current workflow state. Reads "approved".

    Returns:
        The string "finalize" if approved, or "summarize" to regenerate
        the draft.
    """
    return "finalize" if state["approved"] else "summarize"


# ---------------------------------------------------------------------------
# Node 6 - Finalize
# ---------------------------------------------------------------------------

@traceable(name="finalize_agent")
def finalize_agent(state: WorkflowState) -> WorkflowState:
    """Copy the approved draft into the final recommendation field.

    This function only writes "recommendation" when "approved" is True.
    This guard prevents accidental promotion of an unapproved draft,
    even if the graph is rewired.

    Args:
        state: Current workflow state. Reads "approved" and "draft_summary",
            conditionally writes "recommendation".

    Returns:
        The updated state with the final recommendation, if approved.
    """
    if state["approved"]:
        state["recommendation"] = state["draft_summary"]
    return state


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Assemble and compile the LangGraph pipeline.

    Wires all six agent nodes and their edges into a StateGraph, then
    compiles it into an executable application. The research node loops
    via a conditional edge until all plan steps are done. After human
    review, rejection loops back to the summarizer; approval advances
    to finalization and END.

    Returns:
        A compiled LangGraph application.
    """
    graph = StateGraph(WorkflowState)  # type: ignore[arg-type]

    graph.add_node("planner", planner_agent)  # type: ignore[arg-type]
    graph.add_node("research", research_agent)  # type: ignore[arg-type]
    graph.add_node("summarize", summarizer_agent)  # type: ignore[arg-type]
    graph.add_node("formatter", formatter_agent)  # type: ignore[arg-type]
    graph.add_node("human_checkpoint", human_checkpoint_agent)  # type: ignore[arg-type]
    graph.add_node("finalize", finalize_agent)  # type: ignore[arg-type]

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research")
    graph.add_conditional_edges(
        "research",
        should_continue_research,
        {"research": "research", "summarize": "summarize"},
    )
    graph.add_edge("summarize", "formatter")
    graph.add_edge("formatter", "human_checkpoint")
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_checkpoint,
        {"finalize": "finalize", "summarize": "summarize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

@traceable(name="run_research")
def run_research(vehicle_problem: str) -> WorkflowState:
    """Run the full research pipeline for a vehicle problem.

    Builds the graph, initializes the state, and executes synchronously until
    the operator approves a recommendation or kills the process.

    Args:
        vehicle_problem: Year/make/model and symptom description, for example,
            "2014 Ford Focus rough idle and intermittent check engine light".

    Returns:
        The final WorkflowState. The "recommendation" key holds the approved
         text or an empty string if nothing was approved.
    """
    app = build_graph()
    result: WorkflowState = app.invoke(_default_state(vehicle_problem))  # type: ignore[assignment]
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and run the research pipeline.

    Accepts an optional positional argument for the vehicle problem.
    Falls back to a default example if none is provided.
    """
    parser = argparse.ArgumentParser(description="Run the Auto Parts Research Agent workflow.")
    parser.add_argument(
        "problem",
        nargs="?",
        default="2014 Ford Focus rough idle and intermittent check engine light",
        help="Vehicle problem description to research.",
    )
    args = parser.parse_args()

    result = run_research(args.problem)

    print("\n===== FINAL RECOMMENDATION =====")
    print(result.get("recommendation") or "No recommendation approved.")


if __name__ == "__main__":
    main()
