"""
Agent 1: Analyzer
Receives pre-formatted conversation text from state.
Backend is responsible for formatting and truncation before calling run_pipeline().
"""
from langchain.chat_models import init_chat_model
from agent_v1.graph.states import AutoTuneState, TopicAnalysis
from agent_v1.prompts.prompts import analyzer_prompt
from langchain_google_genai import ChatGoogleGenerativeAI


def analyzer_agent(state: AutoTuneState) -> dict:
    """
    Agent 1: Analyze pre-formatted disliked conversations → topic list.
    Input is conversations_text (pre-formatted string from backend).
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_output_tokens=1024)

    # conversations_text is already formatted by the backend — use directly
    conversations_text: str = state["conversations_text"]

    prompt = analyzer_prompt(conversations_text)

    result: TopicAnalysis = (
        llm.with_structured_output(TopicAnalysis, method="json_schema")
           .invoke(prompt)
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