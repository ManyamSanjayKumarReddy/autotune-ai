# agent_builder/rag_model.py

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware

from rag_pipeline.vector_store import ChromaVectorStore
from prompts.prompts_v1 import AGENT_SYSTEM_PROMPT

load_dotenv()

chroma_autotune = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'chroma_autotune')
)


@tool
def search_knowledge_base(query: str, collection_name: str) -> str:
    """Search ChromaDB collection for relevant documents."""
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
    # Ollama Qwen 1.5B - weak model, runs locally
    model = ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0.7,
        num_predict=1024
    )

    return create_agent(
        model=model,
        tools=[search_knowledge_base],
        middleware=[
            ToolCallLimitMiddleware(run_limit=10, thread_limit=15, exit_behavior="end"),
            ModelCallLimitMiddleware(run_limit=10, thread_limit=15, exit_behavior="end"),
        ],
        system_prompt=AGENT_SYSTEM_PROMPT
    )


agent = _build_agent()


def get_response(query: str) -> str:
    """Get answer using weak Ollama model with RAG"""
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content