"""
Agent 2: Link Fetcher
Reads topics from Agent 1 → searches web → LLM picks best links → scrapes.
Uses Gemini 2.0 Flash with json_schema structured output.
Uses before_model middleware hook for logging via create_agent pattern.
"""
import time
from langchain.chat_models import init_chat_model
from agent_v1.graph.states import AutoTuneState, LinkBatch
from agent_v1.tools.web_search import search_web, scrape_url
from agent_v1.prompts.prompts import link_fetcher_prompt
from langchain_google_genai import ChatGoogleGenerativeAI


def link_fetcher_agent(state: AutoTuneState) -> dict:
    """
    Agent 2: Topics → DuckDuckGo → LLM curates best links → scrape content.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_output_tokens=4096)


    topics: list[str] = state["topic_analysis"]["topics"]
    query = " ".join(topics[:3])   # focused query — top 3 topics only

    raw_results = search_web(query, max_results=5)
    time.sleep(1)   # rate-limit respect between DDG and Gemini call

    if not raw_results:
        log = f"[Agent2/LinkFetcher] No DDG results for query='{query}'"
        return {
            "link_batch": {"links": []},
            "scraped_content": "",
            "logs": [log]
        }

    results_text = "\n".join(
        f"{i+1}. [{r.get('title', '?')}] {r.get('link', '')}\n   {r.get('snippet', '')}"
        for i, r in enumerate(raw_results)
    )

    prompt = link_fetcher_prompt(topics, results_text)

    link_batch: LinkBatch = (
        llm.with_structured_output(LinkBatch, method="json_schema")
           .invoke(prompt)
    )

    # Scrape each chosen link
    scraped_parts = []
    for link in link_batch.links:
        content = scrape_url(link.url, max_chars=2000)
        if content and not content.startswith("[SCRAPE FAILED"):
            scraped_parts.append(f"[SOURCE: {link.title}]\n{content}")

    scraped_content = "\n\n".join(scraped_parts)

    log = (
        f"[Agent2/LinkFetcher] query='{query}' | "
        f"links={len(link_batch.links)} | scraped_chars={len(scraped_content)}"
    )

    return {
        "link_batch": link_batch.model_dump(),
        "scraped_content": scraped_content,
        "logs": [log]
    }