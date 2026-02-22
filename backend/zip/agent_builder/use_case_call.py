# agent_builder/usecase_call.py

__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from prompts.prompts_v1 import AGENT_SYSTEM_PROMPT

load_dotenv()


def _build_model():
    """Ollama Qwen 1.5B - weak model, runs locally"""
    return ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0.7,
        num_predict=512
    )


model = _build_model()


def get_response(query: str) -> str:
    """Get answer using weak Ollama model - will show failures"""
    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    response = model.invoke(messages)
    return response.content