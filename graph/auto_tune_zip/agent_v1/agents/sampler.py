# ============================================================
# FILE: agent_v1/agents/sampler.py
# ============================================================
"""
Agent 4: Dataset Sampler with CLI-based HITL using LangGraph interrupt().
"""
import uuid
import json
import time

from langgraph.types import interrupt
from rich.console import Console
from rich.table import Table
from rich import box

from agent_v1.graph.states import AutoTuneState, SampleBatch
from agent_v1.prompts.prompts import sampler_prompt

console = Console()


def sampler_agent(state: AutoTuneState) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4,
        max_output_tokens=4096
    )

    questions: list[str] = state["question_set"]["questions"]
    topics: list[str] = state["topic_analysis"]["topics"]
    scraped: str = state["scraped_content"] or ""
    links: list[dict] = state["link_batch"]["links"] if state["link_batch"] else []
    source_url = links[0]["url"] if links else "no-source"
    batch_id: str = state["batch_id"]

    SUB_BATCH = 3
    all_approved_samples: list[dict] = []

    for i in range(0, len(questions), SUB_BATCH):
        q_batch = questions[i:i + SUB_BATCH]

        prompt = sampler_prompt(
            questions=q_batch,
            topics=topics,
            context=scraped[:3000],
            source_url=source_url
        )

        batch_result: SampleBatch = (
            llm.with_structured_output(SampleBatch, method="json_schema")
               .invoke(prompt)
        )

        # Assign stable IDs
        for sample in batch_result.samples:
            sample.id = f"{batch_id}-{uuid.uuid4().hex[:8]}"

        samples_as_dicts = [s.model_dump() for s in batch_result.samples]

        # ── HITL: pause graph, surface samples to CLI runner ─────────────
        decision = interrupt({
            "batch_index": i // SUB_BATCH,
            "samples": samples_as_dicts
        })

        if decision.get("approved"):
            edits: dict = decision.get("edits", {})
            for s in samples_as_dicts:
                if s["id"] in edits:
                    s["answer"] = edits[s["id"]]
                    s["confidence_score"] = 1.0
            all_approved_samples.extend(samples_as_dicts)

        # Rate-limit pause between sub-batches
        if i + SUB_BATCH < len(questions):
            time.sleep(1)

    log = (
        f"[Agent4/Sampler] questions_in={len(questions)} | "
        f"approved_samples={len(all_approved_samples)}"
    )

    return {
        "sample_batch": {"samples": all_approved_samples},
        "hitl_approved_samples": all_approved_samples,
        "hitl_rejected_ids": [],
        "logs": [log]
    }

