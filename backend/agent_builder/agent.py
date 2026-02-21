# agent_builder/agent.py

__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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
    """
    Search a specific ChromaDB collection for relevant documents about agents, LangChain, and AI.

    Args:
        query: The search query string
        collection_name: The collection to search (e.g., 'langchain_docs', 'agent_patterns', 'code_examples')
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
    # Use Qwen2.5-Coder via HuggingFace
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        task="text-generation",
        max_new_tokens=2048,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.1,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )

    model = ChatHuggingFace(llm=llm)

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
    """
    Get answer to questions about building agents, LangChain, and generative AI.

    Args:
        query: Natural language question

    Returns:
        Answer as string
    """
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content