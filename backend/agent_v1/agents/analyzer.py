"""
Agent 1: Analyzer
Receives pre-formatted conversation text from state.
Backend is responsible for formatting and truncation before calling run_pipeline().
"""
import os
from agent_v1.graph.states import AutoTuneState, TopicAnalysis
from agent_v1.prompts.prompts import analyzer_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL")


def analyzer_agent(state: AutoTuneState) -> dict:
    """
    Agent 1: Analyze pre-formatted disliked conversations → topic list.
    Input is conversations_text (pre-formatted string from backend).
    """
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3, max_output_tokens=1024)

    conversations_text: str = state["conversations_text"]
    structured_llm = llm.with_structured_output(TopicAnalysis, method="json_schema")

    result = None
    last_error = None

    # --- Attempt 1: standard prompt ---
    prompt = analyzer_prompt(conversations_text)
    try:
        result = structured_llm.invoke(prompt)
        if result.is_real_issue and not result.topics:
            last_error = "is_real_issue=True but topics list is empty (attempt 1)"
            print(f"[Agent1/Analyzer] Parse attempt 1 failed: {last_error}")
            result = None
    except Exception as e:
        last_error = e
        print(f"[Agent1/Analyzer] Parse attempt 1 failed: {e}")

    # --- Attempt 2: corrective prompt — tell the model exactly what it did wrong ---
    if result is None:
        try:
            corrective_prompt = (
                f"{prompt}\n\n"
                f"CORRECTION NEEDED: You returned is_real_issue=true but left topics as an empty list. "
                f"This is invalid. If the model has a real knowledge gap, you MUST list the specific topics.\n\n"
                f"Look at the conversations again and list 3-5 specific topics the model clearly didn't know. "
                f"For example: 'how transformer attention works', 'Python async/await syntax', etc.\n\n"
                f"Return a TopicAnalysis with a non-empty topics list."
            )
            result = structured_llm.invoke(corrective_prompt)
            if result.is_real_issue and not result.topics:
                last_error = "is_real_issue=True but topics list is empty (attempt 2)"
                print(f"[Agent1/Analyzer] Parse attempt 2 failed: {last_error}")
                result = None
        except Exception as e:
            last_error = e
            print(f"[Agent1/Analyzer] Parse attempt 2 failed: {e}")

    # --- Attempt 3: strip it down to the simplest possible ask ---
    if result is None:
        try:
            minimal_prompt = (
                f"Read these conversations where an AI gave bad answers:\n\n"
                f"{conversations_text}\n\n"
                f"List 3-5 specific topics the AI clearly did not know well enough. "
                f"Be concrete — not 'AI' but 'how neural networks learn from feedback'.\n\n"
                f"Return: is_real_issue=true, a one-sentence reasoning, and your topics list."
            )
            result = structured_llm.invoke(minimal_prompt)
            if result.is_real_issue and not result.topics:
                last_error = "is_real_issue=True but topics list is empty (attempt 3)"
                print(f"[Agent1/Analyzer] Parse attempt 3 failed: {last_error}")
                result = None
        except Exception as e:
            last_error = e
            print(f"[Agent1/Analyzer] Parse attempt 3 failed: {e}")

    # --- All attempts failed: safe fallback, skip pipeline ---
    if result is None:
        print(f"[Agent1/Analyzer] All 3 attempts failed. Skipping pipeline. Last error: {last_error}")
        result = TopicAnalysis(
            is_real_issue=False,
            reasoning=f"Analyzer could not extract topics after 3 attempts: {last_error}",
            topics=[]
        )

    log = (
        f"[Agent1/Analyzer] batch={state['batch_id']} | "
        f"is_real_issue={result.is_real_issue} | topics={result.topics}"
    )

    return {
        "topic_analysis": result.model_dump(),
        "skip": not result.is_real_issue,
        "logs": [log]
    }