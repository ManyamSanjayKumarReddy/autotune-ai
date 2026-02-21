# api/routes/pipeline2.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipeline.ingest_csv import ingest_csv
from db.models import PipelineRun, ChromaIngestion

router = APIRouter()


class Pipeline2Request(BaseModel):
    csv_path: str
    batch_id: str


@router.post("/ingest")
async def run_ingest(request: Pipeline2Request):
    # Check pipeline run exists
    run = await PipelineRun.get_or_none(batch_id=request.batch_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found for this batch_id")

    # Create ingestion record
    ingestion = await ChromaIngestion.create(
        pipeline_run=run,
        batch_id=request.batch_id,
        collection_name=request.batch_id,
        status="pending"
    )

    try:
        result = ingest_csv(
            csv_path=request.csv_path,
            batch_id=request.batch_id
        )

        ingestion.documents_ingested = result["documents_ingested"]
        ingestion.status = result["status"]
        await ingestion.save()

        return {
            "batch_id": request.batch_id,
            "collection_name": result["collection_name"],
            "documents_ingested": result["documents_ingested"],
            "status": result["status"],
            "error": result.get("error")
        }

    except Exception as e:
        ingestion.status = "failed"
        await ingestion.save()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ingestions")
async def get_all_ingestions():
    ingestions = await ChromaIngestion.all().order_by("-created_at")
    return [
        {
            "id": i.id,
            "batch_id": i.batch_id,
            "collection_name": i.collection_name,
            "documents_ingested": i.documents_ingested,
            "status": i.status,
            "created_at": i.created_at.isoformat()
        }
        for i in ingestions
    ]


@router.get("/stats")
async def get_ingestion_stats():
    total = await ChromaIngestion.all().count()
    successful = await ChromaIngestion.filter(status="success").count()
    failed = await ChromaIngestion.filter(status="failed").count()
    total_docs = await ChromaIngestion.all().values_list("documents_ingested", flat=True)

    return {
        "total_ingestions": total,
        "successful": successful,
        "failed": failed,
        "total_documents_ingested": sum(total_docs)
    }