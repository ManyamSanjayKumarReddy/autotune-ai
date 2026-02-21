# agent.py

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware

from rag_pipeline.vector_store import ChromaVectorStore
from prompts.prompts_v1 import PROMPT_AGENT_BETA_STRUCTURED

load_dotenv()

chroma_autotune = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'chroma_autotune')
)


@tool
def search_knowledge_base(query: str, collection_name: str) -> str:
    """
    Search a specific ChromaDB collection for relevant documents.

    Args:
        query:           The search query string.
        collection_name: The collection to search.
    """
    store = ChromaVectorStore(
        persist_dir=chroma_autotune,
        collection_name=collection_name
    )
    results = store.query(query, top_k=5)

    if not results:
        return f"No documents found in collection '{collection_name}'."

    lines = "\n\n".join(r["text"] for r in results).splitlines()
    cleaned = [
        ln for ln in lines
        if not ln.strip().startswith(("---", "===", "|", "# **Examples**"))
        and ln.strip()
    ]
    return "\n".join(cleaned) if cleaned else "Documents found but could not extract content."


def _build_agent():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, timeout=30, max_tokens=1000)

    return create_agent(
        model=model,
        tools=[search_knowledge_base],
        middleware=[
            ToolCallLimitMiddleware(run_limit=7, thread_limit=10, exit_behavior="end"),
            ModelCallLimitMiddleware(run_limit=7, thread_limit=10, exit_behavior="end"),
        ],
        system_prompt=PROMPT_AGENT_BETA_STRUCTURED
    )

agent = _build_agent()


def get_response(query: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content