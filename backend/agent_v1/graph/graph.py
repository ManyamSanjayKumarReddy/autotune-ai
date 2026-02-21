"""
AutoTune LangGraph — graph assembly and pipeline runner.

Graph flow:
    START → analyzer
               ↓ (skip if no real issue)
           link_fetcher → question_gen → sampler → validator → END

NO HITL - pipeline runs automatically without human intervention.
"""
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

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


# ── Graph builder ──────────────────────────────────────────────────────────

def build_graph():
    """
    Build and compile the AutoTune LangGraph.
    No checkpointer needed since we removed HITL.
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
    # FIXED: Direct edge instead of conditional routing
    graph.add_edge("sampler", "validator")
    graph.add_edge("validator", END)

    return graph.compile()


# ── Public entry point ─────────────────────────────────────────────────────

def run_pipeline(
    conversations_text: str,
    batch_id: str = "batch-001"
) -> dict:
    """
    Run the full AutoTune pipeline without human intervention.

    Args:
        conversations_text: Pre-formatted conversation string from backend.
        batch_id: Unique run identifier.

    Returns:
        Final state dict.
    """
    init_environment()

    graph = build_graph()

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

    # Run pipeline to completion
    result = graph.invoke(initial_state)

    # Export and metrics if we have final output
    if result.get("final_samples"):
        csv_path = export_to_csv(result["final_samples"], batch_id)
        print_metrics(result, csv_path)
    else:
        print("\n[Pipeline] No final samples produced.")
        # Still print logs so you can see what happened
        for log in result.get("logs", []):
            print(f"  {log}")

    return result