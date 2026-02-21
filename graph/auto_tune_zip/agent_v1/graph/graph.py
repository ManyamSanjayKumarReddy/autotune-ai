
"""
AutoTune LangGraph — graph assembly and pipeline runner.

Graph flow:
    START → analyzer
               ↓ (skip if no real issue)
           link_fetcher → question_gen → sampler → validator → END

HITL lives inside sampler_agent via LangGraph interrupt().
The graph requires a MemorySaver checkpointer for interrupt() to work.
"""
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_v1.graph.states import AutoTuneState
from agent_v1.agents.analyzer import analyzer_agent
from agent_v1.agents.link_fetcher import link_fetcher_agent
from agent_v1.agents.question_gen import question_gen_agent
from agent_v1.agents.sampler import sampler_agent
from agent_v1.agents.validator import validator_agent
from agent_v1.export.csv_writer import export_to_csv, print_metrics


def init_environment() -> None:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    load_dotenv()


# ── Routing functions ──────────────────────────────────────────────────────

def route_after_analyzer(state: AutoTuneState) -> str:
    """Skip rest of pipeline if Agent 1 finds no real knowledge gap."""
    return "skip" if state.get("skip") else "link_fetcher"


def route_after_sampler(state: AutoTuneState) -> str:
    """Skip validation if human rejected all samples at HITL."""
    approved = state.get("hitl_approved_samples") or []
    return "skip" if not approved else "validator"


# ── Graph builder ──────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build and compile the AutoTune LangGraph.
    checkpointer must be provided (MemorySaver) for interrupt() to work.
    """
    graph = StateGraph(AutoTuneState)

    graph.add_node("analyzer", analyzer_agent)
    graph.add_node("link_fetcher", link_fetcher_agent)
    graph.add_node("question_gen", question_gen_agent)
    graph.add_node("sampler", sampler_agent)
    graph.add_node("validator", validator_agent)

    graph.add_edge(START, "analyzer")

    graph.add_conditional_edges(
        "analyzer",
        route_after_analyzer,
        {"link_fetcher": "link_fetcher", "skip": END}
    )

    graph.add_edge("link_fetcher", "question_gen")
    graph.add_edge("question_gen", "sampler")

    graph.add_conditional_edges(
        "sampler",
        route_after_sampler,
        {"validator": "validator", "skip": END}
    )

    graph.add_edge("validator", END)

    return graph.compile(checkpointer=checkpointer)


# ── Public entry point ─────────────────────────────────────────────────────

def run_pipeline(
    conversations_text: str,
    batch_id: str = "batch-001"
) -> dict:
    """
    Run the full AutoTune pipeline.

    Args:
        conversations_text: Pre-formatted conversation string from backend.
        batch_id: Unique run identifier.

    Returns:
        Final state dict.
    """
    init_environment()

    # MemorySaver is required — interrupt() checkpoints state here
    graph = build_graph(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": batch_id}}

    initial_state: AutoTuneState = {
        "conversations_text": conversations_text,
        "batch_id": batch_id,
        "topic_analysis": None,
        "link_batch": None,
        "scraped_content": None,
        "question_set": None,
        "sample_batch": None,
        "hitl_approved_samples": None,
        "hitl_rejected_ids": None,
        "validation_report": None,
        "final_samples": None,
        "logs": [],
        "skip": False,
    }

    # Phase 1: run until first interrupt (or END if pipeline skips)
    result = graph.invoke(initial_state, config=config)

    # Phase 2: handle interrupt loop — one pause per sub-batch of 3 questions
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        samples = payload["samples"]
        batch_index = payload.get("batch_index", 0)

        decision = _cli_hitl_review(samples, batch_index)

        # Resume graph with human decision
        result = graph.invoke(
            Command(resume=decision),
            config=config
        )

    # Phase 3: export and metrics if we have final output
    if result.get("final_samples"):
        csv_path = export_to_csv(result["final_samples"], batch_id)
        print_metrics(result, csv_path)
    else:
        print("\n[Pipeline] No final samples produced.")
        # Still print logs so you can see what happened
        for log in result.get("logs", []):
            print(f"  {log}")

    return result


# ── CLI HITL handler ───────────────────────────────────────────────────────

def _cli_hitl_review(samples: list[dict], batch_index: int) -> dict:
    """
    CLI handler for HITL interrupt.
    Displays samples in a rich table and collects approve/reject/edit decision.
    Returns decision dict consumed by sampler_agent after Command(resume=...).

    In Cut 2 this is replaced by a REST endpoint — sampler_agent is unchanged.
    """
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.rule(f"[bold cyan]HITL Review — Batch {batch_index + 1}[/bold cyan]")
    console.print(f"[yellow]{len(samples)} samples generated. Review before saving.[/yellow]\n")

    # Full detail view
    for i, s in enumerate(samples):
        console.print(f"[bold cyan]Sample {i+1}[/bold cyan] [dim]({s['id']})[/dim]")
        console.print(f"  [cyan]Q:[/cyan] {s['question']}")
        console.print(f"  [green]A:[/green] {s['answer']}")
        console.print(f"  [yellow]Score:[/yellow] {s['confidence_score']:.2f}  [dim]Topic: {s['topic']}[/dim]")
        console.print()

    mode = console.input(
        "[bold](A)pprove all / (R)eject all / (E)dit answers → [/bold]"
    ).strip().upper()

    if mode == "A":
        console.print(f"[green]✓ Approved all {len(samples)} samples.[/green]")
        return {"approved": True, "edits": {}}

    elif mode == "R":
        console.print(f"[red]✗ Rejected all {len(samples)} samples.[/red]")
        return {"approved": False, "edits": {}}

    else:  # Edit mode
        edits = {}
        for s in samples:
            console.print(f"\n[cyan]Q:[/cyan] {s['question']}")
            console.print(f"[green]Current A:[/green] {s['answer']}")
            new_ans = console.input(
                "[bold]New answer (press Enter to keep current): [/bold]"
            ).strip()
            if new_ans:
                edits[s["id"]] = new_ans
                console.print("[green]✓ Updated[/green]")
            else:
                console.print("[dim]Kept original[/dim]")

        console.print(f"\n[green]✓ Approved with {len(edits)} edit(s).[/green]")
        return {"approved": True, "edits": edits}