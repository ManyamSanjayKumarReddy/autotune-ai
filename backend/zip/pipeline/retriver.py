# pipeline/retriever.py

from rag_pipeline.vector_store import ChromaVectorStore


def retrieve(query: str, batch_id: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve top-k relevant documents from the batch's Chroma collection.

    Args:
        query:    The original failing query.
        batch_id: Collection name — same batch_id used during ingestion.
        top_k:    Number of docs to retrieve.

    Returns:
        List of dicts with 'text', 'id', 'distance', 'metadata'.
    """
    store = ChromaVectorStore(collection_name=batch_id)
    results = store.query(query, top_k=top_k)

    print(f"[Retriever] '{batch_id}' → {len(results)} docs retrieved")
    return results