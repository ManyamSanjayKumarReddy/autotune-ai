# api/routes/test.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agent_builder.rag_model import get_response as agent_response
from agent_builder.use_case_call import get_response as simple_response

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


# ============== ENDPOINT 1: Simple LLM (No RAG, No Agent) ==============

@router.post("/simple")
async def simple_llm(request: QueryRequest):
    """
    Simple LLM response without RAG or agent framework.
    Uses only the model's training knowledge.
    """
    try:
        response = simple_response(request.query)

        return {
            "query": request.query,
            "response": response,
            "mode": "simple_llm",
            "uses_rag": False,
            "uses_agent": False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== ENDPOINT 2: Agent with RAG ==============

@router.post("/agent")
async def agent_with_rag(request: QueryRequest):
    """
    Agent response with RAG search.
    Searches knowledge base and uses tools.
    """
    try:
        response = agent_response(request.query)

        return {
            "query": request.query,
            "response": response,
            "mode": "agent_with_rag",
            "uses_rag": True,
            "uses_agent": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== ENDPOINT 3: Compare Both ==============

@router.post("/compare")
async def compare_responses(request: QueryRequest):
    """
    Compare simple LLM vs agent with RAG side-by-side.
    """
    try:
        # Get both responses
        simple_resp = simple_response(request.query)
        agent_resp = agent_response(request.query)

        return {
            "query": request.query,
            "simple_llm": {
                "response": simple_resp,
                "mode": "simple_llm",
                "uses_rag": False
            },
            "agent_rag": {
                "response": agent_resp,
                "mode": "agent_with_rag",
                "uses_rag": True
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))