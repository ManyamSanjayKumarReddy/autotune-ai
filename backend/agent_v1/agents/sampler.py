"""
Agent 4: Dataset Sampler - generates training samples from questions.
"""
import uuid
import time
import traceback

from rich.console import Console

from agent_v1.graph.states import AutoTuneState, SampleBatch, DataSample
from agent_v1.prompts.prompts import sampler_prompt
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL")
console = Console()


def _parse_partial_samples(raw_result: SampleBatch, topics: list[str], source_url: str, batch_id: str) -> list[dict]:
    """
    Extract valid samples from a potentially partial result.
    Skips incomplete samples, backfills missing fields on valid ones.
    """
    valid = []
    for sample in raw_result.samples:
        # A sample is only usable if it has the two fields the LLM always generates
        if not sample.question or not sample.answer:
            continue

        # Backfill fields the LLM commonly omits
        sample.id = f"{batch_id}-{uuid.uuid4().hex[:8]}"
        if not sample.topic:
            sample.topic = topics[0] if topics else "general"
        if not sample.source_url:
            sample.source_url = source_url
        if sample.confidence_score == 0.0:
            sample.confidence_score = 0.7

        valid.append(sample.model_dump())
    return valid


def _invoke_sub_batch(structured_llm, prompt: str, sub_batch_num: int,
                      topics: list[str], source_url: str, batch_id: str) -> list[dict]:
    """
    Try to get valid samples from one sub-batch. Returns whatever valid samples we get.
    """
    # Attempt 1
    try:
        result = structured_llm.invoke(prompt)
        samples = _parse_partial_samples(result, topics, source_url, batch_id)
        if samples:
            return samples
        raise ValueError("No valid samples after parsing")
    except Exception as e:
        console.print(f"[yellow]Sub-batch {sub_batch_num} attempt 1 failed: {e}[/yellow]")

    # Attempt 2 — corrective prompt
    try:
        corrective = (
            f"{prompt}\n\n"
            f"CORRECTION: Your previous response had incomplete or missing fields. "
            f"Every sample object MUST have exactly these fields, all filled in:\n"
            f'  "id": "" (leave as empty string)\n'
            f'  "question": the exact question from the list above\n'
            f'  "answer": a complete, detailed answer (minimum 3 sentences)\n'
            f'  "confidence_score": a number like 0.85\n'
            f'  "topic": one of: {", ".join(topics)}\n'
            f'  "source_url": "{source_url}"\n\n'
            f"Generate one complete object per question. Do not truncate. Respond only with valid JSON."
        )
        result = structured_llm.invoke(corrective)
        samples = _parse_partial_samples(result, topics, source_url, batch_id)
        if samples:
            return samples
        raise ValueError("No valid samples after correction")
    except Exception as e:
        console.print(f"[yellow]Sub-batch {sub_batch_num} attempt 2 failed: {e}[/yellow]")

    return []


def sampler_agent(state: AutoTuneState) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI

    try:
        console.print("[cyan]Starting sampler_agent...[/cyan]")

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
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

        structured_llm = llm.with_structured_output(SampleBatch, method="json_mode")

        SUB_BATCH = 3
        all_samples: list[dict] = []

        for i in range(0, len(questions), SUB_BATCH):
            q_batch = questions[i:i + SUB_BATCH]
            sub_batch_num = i // SUB_BATCH + 1

            console.print(f"[cyan]Sub-batch {sub_batch_num}: {len(q_batch)} questions[/cyan]")

            prompt = sampler_prompt(
                questions=q_batch,
                topics=topics,
                context=scraped[:3000],
                source_url=source_url
            )

            samples = _invoke_sub_batch(structured_llm, prompt, sub_batch_num, topics, source_url, batch_id)

            if samples:
                console.print(f"[green]Generated {len(samples)} samples[/green]")
                all_samples.extend(samples)
            else:
                console.print(f"[red]Sub-batch {sub_batch_num} failed after retries — skipping[/red]")

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

        return {
            "sample_batch": {"samples": []},
            "logs": [error_log]
        }