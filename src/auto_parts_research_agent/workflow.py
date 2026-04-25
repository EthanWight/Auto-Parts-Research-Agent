from __future__ import annotations

import argparse
import json
import os
from typing import Literal, TypedDict

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict):
    vehicle_problem: str
    plan: list[str]
    current_step: int
    findings: list[str]
    raw_search_notes: list[str]
    draft_summary: str
    approved: bool
    recommendation: str


def _default_state(vehicle_problem: str) -> WorkflowState:
    return {
        "vehicle_problem": vehicle_problem,
        "plan": [],
        "current_step": 0,
        "findings": [],
        "raw_search_notes": [],
        "draft_summary": "",
        "approved": False,
        "recommendation": "",
    }


def _maybe_llm() -> ChatOpenAI | None:
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    return None


def planner_agent(state: WorkflowState) -> WorkflowState:
    llm = _maybe_llm()
    problem = state["vehicle_problem"]

    if llm is None:
        plan = [
            "Identify likely systems/components tied to the symptom.",
            "Search common failure patterns, TSB/recall hints, and diagnostics.",
            "Map candidate replacement parts and decision criteria.",
        ]
    else:
        prompt = (
            "You are an automotive diagnostics planner for an auto-parts research workflow. "
            "Return ONLY JSON with key 'plan' as an array of 3-5 concise steps. "
            "Focus on diagnosis + parts recommendation."
        )
        response = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=problem)])
        try:
            payload = json.loads(response.content)
            plan = payload["plan"]
        except Exception:
            plan = [line.strip("- ") for line in str(response.content).splitlines() if line.strip()][:4]

    state["plan"] = plan
    state["current_step"] = 0
    return state


def research_agent(state: WorkflowState) -> WorkflowState:
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

    note = f"Step {step_index + 1}: {step}\nResearch Notes: {results}"
    state["raw_search_notes"].append(note)
    state["findings"].append(f"Completed research step {step_index + 1}: {step}")
    state["current_step"] += 1
    return state


def should_continue_research(state: WorkflowState) -> Literal["research", "summarize"]:
    if state["current_step"] < len(state["plan"]):
        return "research"
    return "summarize"


def summarizer_agent(state: WorkflowState) -> WorkflowState:
    llm = _maybe_llm()

    if llm is None:
        top_notes = "\n\n".join(state["raw_search_notes"][:3])
        summary = (
            "Draft recommendation (fallback mode):\n"
            f"Problem: {state['vehicle_problem']}\n"
            "Likely next actions: verify stored OBD-II codes, inspect affected subsystem wiring/connectors, "
            "and compare OEM-spec replacement options from Dorman-compatible catalogs.\n"
            f"Evidence gathered:\n{top_notes}"
        )
    else:
        prompt = (
            "You are a senior parts advisor. Build a concise recommendation with sections: "
            "Likely Cause, Validation Steps, Suggested Part Categories, and Risk/Confidence."
        )
        evidence = "\n\n".join(state["raw_search_notes"])
        response = llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Problem:\n{state['vehicle_problem']}\n\nEvidence:\n{evidence}"),
            ]
        )
        summary = str(response.content)

    state["draft_summary"] = summary
    return state


def human_checkpoint_agent(state: WorkflowState) -> WorkflowState:
    print("\n========== HUMAN CHECKPOINT ==========")
    print(state["draft_summary"])
    print("=====================================\n")
    answer = input("Approve this recommendation draft? [y/N]: ").strip().lower()
    state["approved"] = answer in {"y", "yes"}
    return state


def route_after_checkpoint(state: WorkflowState) -> Literal["finalize", "summarize"]:
    return "finalize" if state["approved"] else "summarize"


def finalize_agent(state: WorkflowState) -> WorkflowState:
    if state["approved"]:
        state["recommendation"] = state["draft_summary"]
    return state


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("planner", planner_agent)
    graph.add_node("research", research_agent)
    graph.add_node("summarize", summarizer_agent)
    graph.add_node("human_checkpoint", human_checkpoint_agent)
    graph.add_node("finalize", finalize_agent)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research")
    graph.add_conditional_edges("research", should_continue_research, {"research": "research", "summarize": "summarize"})
    graph.add_edge("summarize", "human_checkpoint")
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_checkpoint,
        {"finalize": "finalize", "summarize": "summarize"},
    )
    graph.add_edge("finalize", END)

    return graph.compile()


def run_research(vehicle_problem: str) -> WorkflowState:
    app = build_graph()
    return app.invoke(_default_state(vehicle_problem))


def main() -> None:
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
