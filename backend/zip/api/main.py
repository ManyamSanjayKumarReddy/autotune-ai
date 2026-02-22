# api/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tortoise import Tortoise
from dotenv import load_dotenv
import os

from api.routes import pipeline1, pipeline2, test, chatbot

# ── Load Environment Variables ────────────────────────────────
load_dotenv()


# ── Application Lifespan (Startup + Shutdown) ─────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    await Tortoise.init(
        db_url=database_url,
        modules={"models": ["db.models"]},
    )

    # Do NOT generate schemas because Aerich handles migrations
    # await Tortoise.generate_schemas()

    yield

    # ── Shutdown ──────────────────────────────────────────────
    await Tortoise.close_connections()


# ── FastAPI App Instance ──────────────────────────────────────
app = FastAPI(
    title="AutoTune AI",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Configuration ────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "https://auto-tune-scribe.lovable.app",
        "https://autotuneai.theskilledguru.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(pipeline1.router, prefix="/pipeline1", tags=["Pipeline 1"])
app.include_router(pipeline2.router, prefix="/pipeline2", tags=["Pipeline 2"])
app.include_router(test.router, prefix="/test", tags=["Test Results"])
app.include_router(chatbot.router, prefix="/chatbot", tags=["Chatbot"])



# ── Health Endpoint ───────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Connectivity Check Endpoint ───────────────────────────────
@app.get("/connectivity")
async def connectivity():
    checks = {}

    checks["google_api_key"] = "set" if os.getenv("GOOGLE_API_KEY") else "missing"
    checks["database_url"] = "set" if os.getenv("DATABASE_URL") else "missing"
    checks["outputs_dir"] = "ok" if os.path.exists("outputs") else "missing"
    checks["chroma_db"] = (
        "ok" if os.path.exists("chroma_autotune") else "not_yet_created"
    )

    # ── Pipeline 1 Import Check ───────────────────────────────
    try:
        from agent_v1.graph.graph import run_pipeline
        checks["pipeline1"] = "ok"
    except Exception as e:
        checks["pipeline1"] = f"error: {str(e)}"

    # ── Pipeline 2 Import Check ───────────────────────────────
    try:
        from pipeline.ingest_csv import ingest_csv
        checks["pipeline2"] = "ok"
    except Exception as e:
        checks["pipeline2"] = f"error: {str(e)}"

    # ── Database Connectivity Check ───────────────────────────
    try:
        from db.models import PipelineRun
        await PipelineRun.all().count()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"

    overall = (
        "ok"
        if all(v in ("ok", "set", "not_yet_created") for v in checks.values())
        else "degraded"
    )

    return {"status": overall, "checks": checks}