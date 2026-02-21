"""
Agent 5: Validator
Final quality pass on HITL-approved samples.
Uses Gemini 2.0 Flash with json_schema structured output.
"""
import time
from langchain.chat_models import init_chat_model
from agent_v1.graph.states import AutoTuneState, ValidationReport, DataSample
from agent_v1.prompts.prompts import validator_prompt
from langchain_google_genai import ChatGoogleGenerativeAI


def validator_agent(state: AutoTuneState) -> dict:
    """
    Agent 5: Validate approved samples → final clean dataset.
    Processes in sub-batches of 3 to control token usage.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_output_tokens=2048)


    approved: list[dict] = state["hitl_approved_samples"] or []

    if not approved:
        report = ValidationReport(results=[], total=0, passed=0, failed=0)
        return {
            "validation_report": report.model_dump(),
            "final_samples": [],
            "logs": ["[Agent5/Validator] No approved samples — skipping."]
        }

    SUB_BATCH = 3
    all_results = []

    for i in range(0, len(approved), SUB_BATCH):
        batch_dicts = approved[i:i + SUB_BATCH]
        batch = [DataSample(**d) for d in batch_dicts]

        prompt = validator_prompt(batch)

        chunk: ValidationReport = (
            llm.with_structured_output(ValidationReport, method="json_schema")
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