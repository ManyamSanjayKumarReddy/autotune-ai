# api/routes/test.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_pipeline.vector_store import ChromaVectorStore
from db.models import ChromaIngestion

router = APIRouter()


class TestRequest(BaseModel):
    query: str
    batch_id: str


@router.post("/compare")
async def compare(request: TestRequest):
    # Check ingestion exists for this batch
    ingestion = await ChromaIngestion.get_or_none(
        batch_id=request.batch_id,
        status="success"
    )
    if not ingestion:
        raise HTTPException(
            status_code=404,
            detail="No successful ingestion found for this batch_id"
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            max_output_tokens=2048
        )

        # Non-RAG
        non_rag_out = llm.invoke(
            f"You are a helpful assistant.\n\nQuestion: {request.query}\n\nAnswer:"
        )
        non_rag_response = non_rag_out.content

        # RAG
        store = ChromaVectorStore(collection_name=request.batch_id)
        retrieved = store.query(request.query, top_k=3)

        context = (
            "\n\n".join(f"[{i+1}] {doc.get('text', '')}" for i, doc in enumerate(retrieved))
            if retrieved else "No relevant documents found."
        )

        rag_out = llm.invoke(
            f"You are a helpful assistant.\n\n"
            f"Use ONLY the reference below to answer.\n\n"
            f"REFERENCE:\n{context}\n\n"
            f"Question: {request.query}\n\nAnswer:"
        )
        rag_response = rag_out.content

        return {
            "query": request.query,
            "batch_id": request.batch_id,
            "non_rag": {"response": non_rag_response},
            "rag": {
                "response": rag_response,
                "docs_retrieved": len(retrieved),
                "context_used": context
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))