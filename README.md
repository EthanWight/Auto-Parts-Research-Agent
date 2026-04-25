# Auto Parts Research Agent (LangGraph Demo)

This project is a compact **multi-agent workflow** using **LangGraph** to demonstrate planner/executor/supervisor patterns for an automotive use case.

## What it does

Given a vehicle symptom description, the workflow:

1. **Planner Agent** creates a short investigation plan.
2. **Research Agent** executes each plan step with a search tool.
3. **Summarizer Agent** drafts a recommendation.
4. **Human-in-the-loop Checkpoint** asks for approval.
5. **Finalize Agent** publishes the approved recommendation.

Shared graph state carries the problem statement, plan, search notes, and approval status across all nodes.

## Architecture

```text
START
  -> planner
  -> research (loops until all plan steps are complete)
  -> summarize
  -> human_checkpoint
      -> finalize (if approved)
      -> summarize (if not approved, regenerate)
  -> END
```

## Quick start

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
pip install -e .
```

Run:

```bash
auto-parts-agent "2012 Honda Accord P0420 code and sulfur smell"
```

Or:

```bash
python -m auto_parts_research_agent.workflow "2009 Camry overheating at idle"
```

## LLM configuration

- If `OPENAI_API_KEY` is set, planner + summarizer use `ChatOpenAI` (`OPENAI_MODEL`, default `gpt-4o-mini`).
- If no API key is present, the app still runs in deterministic fallback mode to demonstrate graph orchestration.

## Why this is relevant

It explicitly demonstrates:

- Multi-agent orchestration with **LangGraph**.
- Planner / executor / supervisor-style flow.
- Tool invocation (DuckDuckGo search tool).
- Shared, typed state between agents.
- Human approval checkpoint before final output.
