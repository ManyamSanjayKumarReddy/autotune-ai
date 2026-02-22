# agent_v1/agents/validator.py
"""
Agent 5: Validator
Final quality pass on approved samples.
"""
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from agent_v1.graph.states import AutoTuneState, ValidationReport, DataSample
from agent_v1.prompts.prompts import validator_prompt


def validator_agent(state: AutoTuneState) -> dict:
    """
    Agent 5: Validate approved samples → final clean dataset.
    Processes in sub-batches of 3 to control token usage.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=4096
    )

    # Changed from hitl_approved_samples to sample_batch
    approved: list[dict] = state.get("sample_batch", {}).get("samples", [])

    if not approved:
        report = ValidationReport(results=[], total=0, passed=0, failed=0)
        return {
            "validation_report": report.model_dump(),
            "final_samples": [],
            "logs": ["[Agent5/Validator] No samples to validate — skipping."]
        }

    SUB_BATCH = 3
    all_results = []

    for i in range(0, len(approved), SUB_BATCH):
        batch_dicts = approved[i:i + SUB_BATCH]
        batch = [DataSample(**d) for d in batch_dicts]

        prompt = validator_prompt(batch, len(approved))

        # FIXED: Changed to json_mode
        chunk: ValidationReport = (
            llm.with_structured_output(ValidationReport, method="json_mode", include_raw=False)
               .invoke(prompt)
        )
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
        passed=len(passed_ids),
        failed=len(approved) - len(passed_ids)
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