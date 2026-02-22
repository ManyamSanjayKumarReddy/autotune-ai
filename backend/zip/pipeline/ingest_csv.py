# pipeline/ingest_csv.py

from rag_pipeline.vector_store import ChromaVectorStore
import os
import csv
from langchain_core.documents import Document


def ingest_csv(csv_path: str, batch_id: str) -> dict:
    """
    Reads the approved CSV from Pipeline 1 and stores it in ChromaDB.
    collection_name = batch_id  →  each run is isolated.

    Args:
        csv_path:  Path to the CSV file from Pipeline 1.
        batch_id:  Unique run ID — becomes the Chroma collection name.
    """
    result = {
        "collection_name": batch_id,
        "documents_ingested": 0,
        "status": "failed",
        "error": None
    }

    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        documents = _csv_to_documents(csv_path, batch_id)

        if not documents:
            raise ValueError("CSV has no valid rows.")

        store = ChromaVectorStore(collection_name=batch_id)
        store.build_documents(documents)

        result["documents_ingested"] = len(documents)
        result["status"] = "success"
        print(f"[Ingest] ✓ {len(documents)} docs → collection '{batch_id}'")

    except Exception as e:
        result["error"] = str(e)
        print(f"[Ingest] ✗ {e}")

    return result


def _csv_to_documents(csv_path: str, batch_id: str) -> list[Document]:
    documents = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "").strip()
            answer = row.get("answer", "").strip()
            confidence = float(row.get("confidence_score", 0.0))

            # ── Quality gates — skip junk rows ──────────────
            if not question or not answer:
                continue
            if len(answer) < 50:          # too short to be useful
                continue
            if confidence < 0.4:          # below validator threshold
                continue

            page_content = (
                f"Topic: {row.get('topic', '')}\n"
                f"Question: {question}\n"
                f"Answer: {answer}"
            )

            documents.append(Document(
                page_content=page_content,
                metadata={
                    "id": row.get("id", ""),
                    "question": question,
                    "answer": answer,
                    "confidence_score": confidence,
                    "topic": row.get("topic", ""),
                    "source_url": row.get("source_url", ""),
                    "batch_id": batch_id,
                }
            ))

    print(f"[Ingest] {len(documents)} clean rows loaded from CSV")
    return documents