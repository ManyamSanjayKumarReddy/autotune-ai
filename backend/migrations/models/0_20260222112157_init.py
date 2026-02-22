from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "chat_sessions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "session_id" VARCHAR(50) NOT NULL UNIQUE,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "is_active" BOOL NOT NULL  DEFAULT True
);
CREATE INDEX IF NOT EXISTS "idx_chat_sessio_session_a5c17d" ON "chat_sessions" ("session_id");
COMMENT ON TABLE "chat_sessions" IS 'Chat session - represents a conversation thread';
CREATE TABLE IF NOT EXISTS "chat_messages" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "message_id" VARCHAR(50) NOT NULL UNIQUE,
    "role" VARCHAR(20) NOT NULL,
    "content" TEXT NOT NULL,
    "parent_message_id" VARCHAR(50),
    "liked" BOOL,
    "model_used" VARCHAR(100),
    "uses_rag" BOOL NOT NULL  DEFAULT False,
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "session_id" INT NOT NULL REFERENCES "chat_sessions" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_chat_messag_message_7a269e" ON "chat_messages" ("message_id");
COMMENT ON TABLE "chat_messages" IS 'Individual chat messages';
CREATE TABLE IF NOT EXISTS "pipeline_runs" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "batch_id" VARCHAR(255) NOT NULL UNIQUE,
    "conversations_text" TEXT NOT NULL,
    "csv_path" VARCHAR(500),
    "final_samples_count" INT NOT NULL  DEFAULT 0,
    "status" VARCHAR(50) NOT NULL  DEFAULT 'pending',
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS "chroma_ingestions" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "batch_id" VARCHAR(255) NOT NULL,
    "collection_name" VARCHAR(255) NOT NULL,
    "documents_ingested" INT NOT NULL  DEFAULT 0,
    "status" VARCHAR(50) NOT NULL  DEFAULT 'pending',
    "created_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
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
