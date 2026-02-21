"""
Web search and scraping tools.

v1 Pattern used:
  - @tool decorator from langchain.tools
  - ToolRuntime for runtime context (batch_id for logging)
  - DuckDuckGoSearchResults from langchain-community with output_format="list"
"""
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain_community.tools import DuckDuckGoSearchResults
import requests
from bs4 import BeautifulSoup
import time

# Shared DDG instance — avoids re-instantiation on every call
_ddg = DuckDuckGoSearchResults(output_format="list")


# Runtime context schema — injected into tools via ToolRuntime
@dataclass
class SearchContext:
    batch_id: str = "default"


# ── Raw helper functions (non-tool, used directly by agents) ───────────────

def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Returns list of {title, link, snippet} dicts via DuckDuckGo.
    No API key required. Rate-limit safe with built-in retry.
    """
    for attempt in range(3):
        try:
            results = _ddg.invoke(query)
            return results[:max_results]
        except Exception as e:
            if attempt == 2:
                print(f"[web_search] DDG failed after 3 attempts: {e}")
                return []
            time.sleep(2 * (attempt + 1))
    return []


def scrape_url(url: str, max_chars: int = 3000) -> str:
    """
    Fetches a URL and returns cleaned plain text capped at max_chars.
    Never raises — returns a failure message string on any error.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (AutoTune AI research bot)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    except Exception as e:
        return f"[SCRAPE FAILED: {e}]"


# ── @tool wrappers (used by create_agent in sampler) ─────────────────────

@tool
def web_search_tool(query: str, runtime: ToolRuntime[SearchContext]) -> str:
    """
    Search the web for a given query. Returns top 5 results as formatted text.
    Use this to find authoritative sources on training topics.

    Args:
        query: Search query string (be specific, 3-6 words work best)
    """
    batch_id = runtime.context.batch_id
    results = search_web(query, max_results=5)
    if not results:
        return "No results found."

    print(f"[{batch_id}] web_search_tool: '{query}' → {len(results)} results")

    return "\n".join(
        f"[{r.get('title', '?')}] {r.get('link', '')} — {r.get('snippet', '')}"
        for r in results
    )


@tool
def scrape_url_tool(url: str, runtime: ToolRuntime[SearchContext]) -> str:
    """
    Fetch and extract plain text from a web URL.
    Use this after web_search_tool to get full content from a specific page.
    Returns up to 3000 characters of cleaned text.

    Args:
        url: Full URL to fetch (must start with https://)
    """
    batch_id = runtime.context.batch_id
    content = scrape_url(url, max_chars=3000)
    print(f"[{batch_id}] scrape_url_tool: {url[:60]}... → {len(content)} chars")
    return content