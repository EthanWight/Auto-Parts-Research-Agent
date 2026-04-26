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
import difflib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# Load environment variables from .env before any os.getenv() calls.
load_dotenv()

# Shared Rich console used for all terminal output.
console = Console()


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
        previous_draft: Snapshot of the last draft shown to the operator,
            used to display a diff when a revised version comes back.
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
    previous_draft: str
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
        "previous_draft": "",
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


def _show_diff(old: str, new: str) -> None:
    """Display a colored unified diff between two draft versions.

    Green lines are additions in the new draft, red lines are removals
    from the old draft. Unchanged context lines are shown in dim white.

    Args:
        old: The previous draft text.
        new: The revised draft text.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="Previous Draft", tofile="Revised Draft")

    diff_text = Text()
    has_content = False
    for line in diff:
        has_content = True
        if line.startswith("+") and not line.startswith("+++"):
            diff_text.append(line, style="green")
        elif line.startswith("-") and not line.startswith("---"):
            diff_text.append(line, style="red")
        elif line.startswith("@@"):
            diff_text.append(line, style="cyan")
        else:
            diff_text.append(line, style="dim")

    if has_content:
        console.print(Panel(diff_text, title="Changes From Previous Draft", border_style="yellow"))
    else:
        console.print("[dim]No differences detected.[/dim]")


# ---------------------------------------------------------------------------
# Node 1 - Planner
# ---------------------------------------------------------------------------

@traceable(name="planner_agent")
def planner_agent(state: WorkflowState) -> WorkflowState:
    """Decompose the vehicle problem into 3-5 ordered research steps.

    This function sends the problem to the LLM with a strict "JSON only"
    system prompt. It applies two fallback layers to handle quirky LLM output:
    1. Strips any <think> block, then extracts the first JSON object via regex.
    2. If JSON parsing fails, it splits the response into lines and takes the
       first four non-empty lines as plan steps.

    Args:
        state: Current workflow state. Reads "vehicle_problem", writes "plan"
            and resets "current_step" to 0.

    Returns:
        The updated state with the research plan populated.
    """
    llm = _maybe_llm()
    problem = state["vehicle_problem"]

    console.print("\n[bold cyan]Building research plan...[/bold cyan]")

    prompt = (
        "You are an automotive diagnostics planner for an auto-parts research workflow. "
        "Return ONLY a JSON object with key 'plan' as an array of 3-5 concise steps. "
        "Do not include any explanation or markdown - only valid JSON. "
        "Focus on diagnosis + parts recommendation."
    )

    response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=problem)])
    content = _strip_thinking(str(response.content))

    try:
        match = re.search(r"\{.*}", content, re.DOTALL)
        payload = json.loads(match.group() if match else content)
        plan = payload["plan"]
    except (json.JSONDecodeError, KeyError, AttributeError, TypeError):
        plan = [line.strip("- ") for line in content.splitlines() if line.strip()][:4]

    for i, step in enumerate(plan, 1):
        console.print(f"  [green]{i}.[/green] {step}")

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

    console.print(
        f"  [bold blue]Researching[/bold blue] [{step_index + 1}/{len(plan)}] {step}"
    )

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
    console.print("\n[bold cyan]Generating recommendation draft...[/bold cyan]")

    prompt = (
        "You are a senior parts advisor. Build a concise recommendation with sections: "
        "Likely Cause, Validation Steps, Suggested Part Categories, and Risk/Confidence."
    )

    evidence = "\n\n".join(state["raw_search_notes"])

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
    console.print("[bold cyan]Polishing for readability...[/bold cyan]")

    prompt = (
        "You are a friendly automotive service advisor writing a report for a car owner "
        "who is not a mechanic. Rewrite the following technical recommendation so it is "
        "easy to understand and pleasant to read. Follow these rules:\n"
        "1. Start with a short 2-3 sentence plain-English summary of the bottom line.\n"
        "2. Use bold headings for each section.\n"
        "3. Use bullet points for lists of steps or parts.\n"
        "4. Replace technical jargon with simple, everyday language.\n"
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

    Renders the recommendation as formatted Markdown inside a Rich panel.
    If a previous draft exists (meaning this is a revision cycle), a colored
    diff is shown first so the operator can see exactly what changed.

    The operator types Y/yes/Enter to approve, or anything else to reject.
    On rejection, the operator provides revision notes that get fed back
    into the summarizer on the next pass.

    Args:
        state: Current workflow state. Reads "draft_summary" and
            "previous_draft". Writes "approved", "operator_feedback",
            and "previous_draft".

    Returns:
        The updated state with the approval decision is recorded.
    """
    console.print()

    # If there is a previous draft, show the diff so the user can see
    # exactly what the AI changed based on their feedback.
    if state.get("previous_draft"):
        _show_diff(state["previous_draft"], state["draft_summary"])
        console.print()

    # Render the current draft as formatted Markdown inside a bordered panel.
    console.print(
        Panel(
            Markdown(state["draft_summary"]),
            title="[bold]Draft Recommendation[/bold]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )

    console.print("\n[bold]Approve this recommendation?[/bold] [dim](Y/N)[/dim] ", end="")
    answer = input().strip().lower()

    approved = answer in {"y", "yes", ""}
    state["approved"] = approved

    if not approved:
        # Save the current draft so we can diff against the next revision.
        state["previous_draft"] = state["draft_summary"]

        console.print(
            "\n[yellow]Tell the AI what to change. Be as specific as you like "
            "(e.g., add part numbers, focus on electrical, shorter summary).[/yellow]"
        )
        feedback = Prompt.ask("[bold]Revision notes[/bold]")
        state["operator_feedback"] = feedback
        console.print("\n[dim]Regenerating draft with your feedback...[/dim]")

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

def _gather_vehicle_info() -> str:
    """Prompt the user for vehicle details and symptoms using Rich prompts.

    Asks a series of short questions inside a styled panel and combines
    the answers into a single problem description string.

    Returns:
        A combined problem description string.
    """
    console.print(
        Panel(
            "[bold white]Welcome! Answer a few questions about your vehicle\n"
            "and the issue you are experiencing.[/bold white]",
            title="[bold bright_blue]Auto Parts Research Agent[/bold bright_blue]",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )
    console.print()

    year = Prompt.ask("[bold]Vehicle year[/bold]")
    make = Prompt.ask("[bold]Make[/bold]")
    model = Prompt.ask("[bold]Model[/bold]")
    mileage = Prompt.ask("[bold]Approximate mileage[/bold]")
    console.print()
    symptoms = Prompt.ask("[bold]Describe the issue you are experiencing[/bold]")
    when = Prompt.ask("[bold]When does it happen?[/bold]")
    extras = Prompt.ask("[bold]Any warning lights, codes, or other details?[/bold]")

    parts = [f"{year} {make} {model}"]
    if mileage:
        parts.append(f"with approximately {mileage} miles")
    parts.append(f"is experiencing: {symptoms}.")
    if when:
        parts.append(f"This happens {when}.")
    if extras:
        parts.append(f"Additional details: {extras}.")

    problem = " ".join(parts)

    console.print()
    console.print(
        Panel(problem, title="[bold]Researching[/bold]", border_style="green", padding=(1, 2))
    )
    console.print()

    return problem


def _save_report(result: WorkflowState) -> Path:
    """Save the approved recommendation to a Markdown file in the reports' folder.

    Args:
        result: The final workflow state containing the recommendation.

    Returns:
        The path to the saved report file.
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = reports_dir / f"report_{timestamp}.md"

    now = datetime.now().strftime("%B %d, %Y %I:%M %p")
    content = (
        f"# Auto Parts Research Report\n\n"
        f"**Date:** {now}\n\n"
        f"**Vehicle Problem:** {result['vehicle_problem']}\n\n"
        f"---\n\n"
        f"{result['recommendation']}\n"
    )

    filename.write_text(content, encoding="utf-8")
    return filename


def main() -> None:
    """Run the research pipeline interactively or with a CLI argument.

    If a problem description is passed as a command-line argument, it is
    used directly. Otherwise, the user is prompted with a series of
    questions to gather vehicle details and symptoms.
    """
    parser = argparse.ArgumentParser(description="Run the Auto Parts Research Agent workflow.")
    parser.add_argument(
        "problem",
        nargs="?",
        default=None,
        help="Vehicle problem description. If omitted, you will be prompted.",
    )
    args = parser.parse_args()

    problem = args.problem if args.problem else _gather_vehicle_info()

    result = run_research(problem)

    recommendation = result.get("recommendation", "")

    if recommendation:
        console.print()
        console.print(
            Panel(
                Markdown(recommendation),
                title="[bold green]Final Approved Recommendation[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        filepath = _save_report(result)
        console.print(f"\n[bold green]Report saved to:[/bold green] {filepath}")
    else:
        console.print("\n[bold red]No recommendation was approved.[/bold red]")

    console.print()


if __name__ == "__main__":
    main()
