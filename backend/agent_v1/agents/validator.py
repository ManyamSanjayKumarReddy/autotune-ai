"""
Agent 5: Validator

Two-layer validation:
1) Deterministic validation (schema, duplicates, empty fields, length, confidence, URL)
2) Optional semantic LLM validation (hallucination, coherence, contradictions)

Deterministic layer ALWAYS runs first.
LLM layer only runs on samples that pass deterministic checks.
"""

import time
import hashlib
from urllib.parse import urlparse
from typing import List, Dict

from agent_v1.graph.states import (
    AutoTuneState,
    ValidationReport,
    ValidationResult,
    DataSample,
)

from agent_v1.prompts.prompts import validator_prompt
from langchain_google_genai import ChatGoogleGenerativeAI


# ───────────────────────────────────────────────────────────────
# Deterministic Validation Layer
# ───────────────────────────────────────────────────────────────

MIN_QUESTION_LEN = 15
MIN_ANSWER_LEN = 40
MIN_CONFIDENCE = 0.4


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return bool(parsed.scheme and parsed.netloc)
    except Exception:
        return False


def deterministic_validate(samples: List[Dict]) -> (List[Dict], List[ValidationResult]):
    """
    Returns:
        (valid_samples, validation_results_for_failed)
    """

    seen_questions = set()
    seen_answers = set()

    valid_samples = []
    failures: List[ValidationResult] = []

    for sample in samples:
        sample_id = sample.get("id", "")
        issues = []

        # Required fields check
        required_fields = ["id", "question", "answer", "confidence_score", "topic", "source_url"]
        for field in required_fields:
            if field not in sample or sample[field] is None:
                issues.append(f"Missing field: {field}")

        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        confidence = float(sample.get("confidence_score", 0))
        source_url = str(sample.get("source_url", "")).strip()

        # Empty checks
        if not question:
            issues.append("Empty question")
        if not answer:
            issues.append("Empty answer")

        # Length checks
        if len(question) < MIN_QUESTION_LEN:
            issues.append("Question too short")
        if len(answer) < MIN_ANSWER_LEN:
            issues.append("Answer too short")

        # Confidence threshold
        if confidence < MIN_CONFIDENCE:
            issues.append("Confidence score below threshold")

        # URL validation
        if not _is_valid_url(source_url):
            issues.append("Invalid source URL")

        # Duplicate detection (normalized hash)
        q_hash = _hash(_normalize(question))
        a_hash = _hash(_normalize(answer))

        if q_hash in seen_questions:
            issues.append("Duplicate question detected")
        if a_hash in seen_answers:
            issues.append("Duplicate answer detected")

        # Final decision
        if issues:
            failures.append(
                ValidationResult(
                    sample_id=sample_id,
                    is_valid=False,
                    issues=issues,
                    revised_answer=None,
                )
            )
        else:
            seen_questions.add(q_hash)
            seen_answers.add(a_hash)
            valid_samples.append(sample)

    return valid_samples, failures


# ───────────────────────────────────────────────────────────────
# Semantic LLM Validation Layer
# ───────────────────────────────────────────────────────────────

def semantic_validate(samples: List[Dict]) -> ValidationReport:
    """
    Runs LLM-based semantic validation in sub-batches of 3.
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=2048,
    )

    SUB_BATCH = 3
    all_results = []

    for i in range(0, len(samples), SUB_BATCH):
        batch_dicts = samples[i:i + SUB_BATCH]
        batch = [DataSample(**d) for d in batch_dicts]

        prompt = validator_prompt(batch, len(samples))

        chunk: ValidationReport = (
            llm.with_structured_output(ValidationReport, method="json_mode")
            .invoke(prompt)
        )

        all_results.extend(chunk.results)

        if i + SUB_BATCH < len(samples):
            time.sleep(1)

    passed = [r for r in all_results if r.is_valid]
    failed = [r for r in all_results if not r.is_valid]

    return ValidationReport(
        results=all_results,
        total=len(samples),
        passed=len(passed),
        failed=len(failed),
    )


# ───────────────────────────────────────────────────────────────
# Main Agent Entry
# ───────────────────────────────────────────────────────────────

def validator_agent(state: AutoTuneState) -> dict:
    approved: List[Dict] = state.get("sample_batch", {}).get("samples", [])

    if not approved:
        empty_report = ValidationReport(results=[], total=0, passed=0, failed=0)
        return {
            "validation_report": empty_report.model_dump(),
            "final_samples": [],
            "logs": ["[Validator] No samples to validate — skipping."],
        }

    # 1️⃣ Deterministic validation
    valid_samples, deterministic_failures = deterministic_validate(approved)

    if not valid_samples:
        # Entire dataset rejected deterministically
        report = ValidationReport(
            results=deterministic_failures,
            total=len(approved),
            passed=0,
            failed=len(approved),
        )

        return {
            "validation_report": report.model_dump(),
            "final_samples": [],
            "logs": ["[Validator] All samples rejected by deterministic validation."],
        }

    # 2️⃣ Semantic LLM validation
    semantic_report = semantic_validate(valid_samples)

    passed_ids = {r.sample_id for r in semantic_report.results if r.is_valid}
    revisions = {
        r.sample_id: r.revised_answer
        for r in semantic_report.results
        if r.revised_answer
    }

    final_samples = []
    for sample in valid_samples:
        if sample["id"] in passed_ids:
            if sample["id"] in revisions:
                sample = {**sample, "answer": revisions[sample["id"]]}
            final_samples.append(sample)

    full_results = deterministic_failures + semantic_report.results

    full_report = ValidationReport(
        results=full_results,
        total=len(approved),
        passed=len(final_samples),
        failed=len(approved) - len(final_samples),
    )

    log = (
        f"[Validator] total={full_report.total} | "
        f"passed={full_report.passed} | failed={full_report.failed}"
    )

    return {
        "validation_report": full_report.model_dump(),
        "final_samples": final_samples,
        "logs": [log],
    }