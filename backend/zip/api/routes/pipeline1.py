# api/routes/pipeline1.py

import csv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

from agent_v1.graph.graph import run_pipeline
from db.models import PipelineRun

router = APIRouter()


class Pipeline1Request(BaseModel):
    conversations_text: str
    batch_id: str


@router.post("/run")
async def run_pipeline1(request: Pipeline1Request):
    # Create DB record
    run = await PipelineRun.create(
        batch_id=request.batch_id,
        conversations_text=request.conversations_text,
        status="pending"
    )

    try:
        result = run_pipeline(
            conversations_text=request.conversations_text,
            batch_id=request.batch_id
        )

        csv_path = None
        csv_rows = []
        final_samples = result.get("final_samples") or []

        if final_samples:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"outputs/{request.batch_id}_{timestamp}.csv"

            # Read CSV rows to return to frontend
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)

        # Update DB record
        run.csv_path = csv_path
        run.final_samples_count = len(final_samples)
        run.status = "success" if final_samples else "no_samples"
        await run.save()

        return {
            "batch_id": request.batch_id,
            "status": run.status,
            "csv_path": csv_path,
            "final_samples_count": len(final_samples),
            "csv_rows": csv_rows,
            "logs": result.get("logs", [])
        }

    except Exception as e:
        run.status = "failed"
        await run.save()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
async def get_all_runs():
    runs = await PipelineRun.all().order_by("-created_at")
    return [
        {
            "id": r.id,
            "batch_id": r.batch_id,
            "status": r.status,
            "final_samples_count": r.final_samples_count,
            "csv_path": r.csv_path,
            "created_at": r.created_at.isoformat()
        }
        for r in runs
    ]


@router.get("/runs/{batch_id}")
async def get_run(batch_id: str):
    run = await PipelineRun.get_or_none(batch_id=batch_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    csv_rows = []
    if run.csv_path:
        try:
            with open(run.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
        except FileNotFoundError:
            pass

    return {
        "id": run.id,
        "batch_id": run.batch_id,
        "status": run.status,
        "final_samples_count": run.final_samples_count,
        "csv_path": run.csv_path,
        "csv_rows": csv_rows,
        "created_at": run.created_at.isoformat()
    }


@router.get("/stats")
async def get_stats():
    total_runs = await PipelineRun.all().count()
    successful = await PipelineRun.filter(status="success").count()
    failed = await PipelineRun.filter(status="failed").count()
    no_samples = await PipelineRun.filter(status="no_samples").count()
    total_samples = await PipelineRun.all().values_list("final_samples_count", flat=True)

    return {
        "total_runs": total_runs,
        "successful": successful,
        "failed": failed,
        "no_samples": no_samples,
        "total_samples_generated": sum(total_samples)
    }