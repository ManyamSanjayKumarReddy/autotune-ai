"""
HITL node using LangGraph's interrupt() primitive.

How it works:
  1. Node calls interrupt(payload) — execution pauses, state is checkpointed.
  2. The caller (app.py or FastAPI) receives {"__interrupt__": [...]} in the result.
  3. The caller collects human decisions and re-invokes with Command(resume=decisions).
  4. interrupt() returns the resume value — node continues from that point.

This is the same pattern whether the caller is a CLI or a REST endpoint.
"""
from langgraph.types import interrupt
from agent_v1.graph.states import AutoTuneState


def hitl_review_node(state: AutoTuneState) -> dict:
    """
    Pauses graph execution for human review of generated samples.
    Returns approved samples and rejected IDs.
    """
    samples: list[dict] = state["sample_batch"]["samples"]

    # interrupt() suspends graph here. The value is surfaced to the caller.
    # In CLI mode, app.py reads this and prompts the user.
    # In REST mode, the frontend reads this via GET /rag_pipeline/{batch_id}/review.
    decision: dict = interrupt({
        "action": "review_samples",
        "batch_id": state["batch_id"],
        "total": len(samples),
        "samples": samples   # full sample list for the reviewer
    })

    # decision is whatever the caller passed to Command(resume=...)
    # Expected shape: {"approved_ids": [...], "rejected_ids": [...], "edits": {"id": "new_answer"}}
    approved_ids: set = set(decision.get("approved_ids", []))
    rejected_ids: list = decision.get("rejected_ids", [])
    edits: dict = decision.get("edits", {})       # {sample_id: new_answer_string}

    approved_samples = []
    for sample in samples:
        if sample["id"] in approved_ids:
            if sample["id"] in edits:
                sample = {**sample, "answer": edits[sample["id"]], "confidence_score": 1.0}
            approved_samples.append(sample)

    log = f"[HITL] approved={len(approved_samples)} | rejected={len(rejected_ids)}"

    return {
        "hitl_approved_samples": approved_samples,
        "hitl_rejected_ids": rejected_ids,
        "logs": [log]
    }