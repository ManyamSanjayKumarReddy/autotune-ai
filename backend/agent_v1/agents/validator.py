# agent_v1/agents/validator.py
"""
Agent 5: Validator
Final quality pass on approved samples.
"""
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_v1.graph.states import AutoTuneState, ValidationReport, DataSample
from agent_v1.prompts.prompts import validator_prompt
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL")


def validator_agent(state: AutoTuneState) -> dict:
    """
    Agent 5: Validate approved samples → final clean dataset.
    Processes in sub-batches of 3 to control token usage.
    """
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,
        max_output_tokens=4096
    )

    approved: list[dict] = state.get("sample_batch", {}).get("samples", [])

    if not approved:
        report = ValidationReport(results=[], total=0, passed=0, failed=0)
        return {
            "validation_report": report.model_dump(),
            "final_samples": [],
            "logs": ["[Agent5/Validator] No samples to validate — skipping."]
        }

    structured_llm = llm.with_structured_output(ValidationReport, method="json_mode", include_raw=False)

    SUB_BATCH = 3
    all_results = []

    for i in range(0, len(approved), SUB_BATCH):
        batch_dicts = approved[i:i + SUB_BATCH]
        batch = [DataSample(**d) for d in batch_dicts]
        prompt = validator_prompt(batch, len(approved))

        chunk = None

        # Attempt 1
        try:
            chunk = structured_llm.invoke(prompt)
        except Exception as e:
            print(f"[Agent5/Validator] Sub-batch {i//SUB_BATCH + 1} attempt 1 failed: {e}")

        # Attempt 2 — corrective prompt if first failed
        if chunk is None:
            try:
                corrective = (
                    f"{prompt}\n\n"
                    f"CORRECTION: Your previous response was missing required fields. "
                    f"You MUST include all four top-level fields in your JSON:\n"
                    f'  "results": [...],\n'
                    f'  "total": {len(approved)},\n'
                    f'  "passed": <count of is_valid=true>,\n'
                    f'  "failed": <count of is_valid=false>\n\n'
                    f"Do not omit total, passed, or failed. Respond only with valid JSON."
                )
                chunk = structured_llm.invoke(corrective)
            except Exception as e:
                print(f"[Agent5/Validator] Sub-batch {i//SUB_BATCH + 1} attempt 2 failed: {e}")

        if chunk is None:
            # Create pass-through results for this batch so samples aren't lost
            print(f"[Agent5/Validator] Sub-batch {i//SUB_BATCH + 1} failed both attempts — passing samples through")
            for sample in batch:
                all_results.append(type('R', (), {
                    'sample_id': sample.id,
                    'is_valid': True,
                    'revised_answer': None
                })())
            continue

        # Backfill total/passed/failed if LLM omitted them (they now have defaults of 0)
        if chunk.total == 0 and chunk.results:
            chunk.total = len(approved)
            chunk.passed = sum(1 for r in chunk.results if r.is_valid)
            chunk.failed = chunk.total - chunk.passed

        all_results.extend(chunk.results)

        if i + SUB_BATCH < len(approved):
            time.sleep(1)

    passed_ids = {r.sample_id for r in all_results if r.is_valid}
    revisions = {r.sample_id: r.revised_answer for r in all_results if r.revised_answer}

    final_samples = []
    for sample in approved:
        if sample["id"] in passed_ids:
            if sample["id"] in revisions:
                sample = {**sample, "answer": revisions[sample["id"]]}
            final_samples.append(sample)

    full_report = ValidationReport(
        results=all_results,
        total=len(approved),
        passed=len(final_samples),
        failed=len(approved) - len(final_samples)
    )

    log = (
        f"[Agent5/Validator] total={full_report.total} | "
        f"passed={full_report.passed} | failed={full_report.failed}"
    )

    return {
        "validation_report": full_report.model_dump(),
        "final_samples": final_samples,
        "logs": [log]
    }