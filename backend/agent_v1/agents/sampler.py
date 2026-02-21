"""
Agent 4: Dataset Sampler - generates training samples from questions.
"""
import uuid
import json
import time
import traceback

from rich.console import Console

from agent_v1.graph.states import AutoTuneState, SampleBatch
from agent_v1.prompts.prompts import sampler_prompt

console = Console()


def sampler_agent(state: AutoTuneState) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        console.print("[cyan]Starting sampler_agent...[/cyan]")

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

        console.print(f"[cyan]Processing {len(questions)} questions...[/cyan]")

        SUB_BATCH = 3
        all_samples: list[dict] = []

        for i in range(0, len(questions), SUB_BATCH):
            q_batch = questions[i:i + SUB_BATCH]

            console.print(f"[cyan]Sub-batch {i//SUB_BATCH + 1}: {len(q_batch)} questions[/cyan]")

            prompt = sampler_prompt(
                questions=q_batch,
                topics=topics,
                context=scraped[:3000],
                source_url=source_url
            )

            # FIXED: Changed from json_schema to json_mode
            batch_result: SampleBatch = (
                llm.with_structured_output(SampleBatch, method="json_mode")
                   .invoke(prompt)
            )

            console.print(f"[green]Generated {len(batch_result.samples)} samples[/green]")

            # Assign stable IDs
            for sample in batch_result.samples:
                sample.id = f"{batch_id}-{uuid.uuid4().hex[:8]}"

            samples_as_dicts = [s.model_dump() for s in batch_result.samples]
            all_samples.extend(samples_as_dicts)

            # Rate-limit pause between sub-batches
            if i + SUB_BATCH < len(questions):
                time.sleep(1)

        log = (
            f"[Agent4/Sampler] questions_in={len(questions)} | "
            f"samples_generated={len(all_samples)}"
        )

        console.print(f"[green bold]✓ Sampler completed: {len(all_samples)} samples[/green bold]")

        return {
            "sample_batch": {"samples": all_samples},
            "logs": [log]
        }

    except Exception as e:
        error_log = f"[Agent4/Sampler] ERROR: {str(e)}"
        console.print(f"[red bold]{error_log}[/red bold]")
        console.print(f"[red]{traceback.format_exc()}[/red]")

        # Return empty but valid response so pipeline continues
        return {
            "sample_batch": {"samples": []},
            "logs": [error_log]
        }