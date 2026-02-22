"""
Agent 3: Question Generator
Reads scraped content + topics → generates 10 diverse training questions.
"""
from langchain.chat_models import init_chat_model
from agent_v1.graph.states import AutoTuneState, QuestionSet
from agent_v1.prompts.prompts import question_gen_prompt
from langchain_google_genai import ChatGoogleGenerativeAI


def question_gen_agent(state: AutoTuneState) -> dict:
    """
    Agent 3: Scraped content + topics → 10 training questions.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_output_tokens=4096)


    topics: list[str] = state["topic_analysis"]["topics"]
    scraped: str = state["scraped_content"] or ""
    scraped_trimmed = scraped[:6000]   # ~1500 tokens — safe for Gemini 2.0 Flash

    prompt = question_gen_prompt(topics, scraped_trimmed)

    question_set: QuestionSet = (
        llm.with_structured_output(QuestionSet, method="json_schema")
           .invoke(prompt)
    )

    log = (
        f"[Agent3/QuestionGen] topics={topics} | "
        f"questions_generated={len(question_set.questions)}"
    )

    return {
        "question_set": question_set.model_dump(),
        "logs": [log]
    }