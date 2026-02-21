from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "pipeline_runs" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "batch_id" VARCHAR(255) NOT NULL UNIQUE,
    "conversations_text" TEXT NOT NULL,
    "csv_path" VARCHAR(500),
    "final_samples_count" INT NOT NULL,
    "status" VARCHAR(50) NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS "chroma_ingestions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "batch_id" VARCHAR(255) NOT NULL,
    "collection_name" VARCHAR(255) NOT NULL,
    "documents_ingested" INT NOT NULL,
    "status" VARCHAR(50) NOT NULL,
    "created_at" TIMESTAMPTZ NOT NULL,
    "pipeline_run_id" INT NOT NULL REFERENCES "pipeline_runs" ("id") ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS "aerich" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "version" VARCHAR(255) NOT NULL,
    "app" VARCHAR(100) NOT NULL,
    "content" JSONB NOT NULL
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
