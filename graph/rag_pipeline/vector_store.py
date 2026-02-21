__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import chromadb
import numpy as np
from typing import List, Any
from rag_pipeline.embeddings import EmbeddingPipeline
from rag_pipeline.encoder import get_sentence_transformer
import uuid

chroma_autotune = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chroma_autotune'))

class ChromaVectorStore:

    def __init__(
        self,
        persist_dir: str = chroma_autotune,
        embedding_model: str = 'all-MiniLM-L6-v2',
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "autotune_default"
    ):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name

        self.client = None
        self.collection = None

        self._initilize_store()

    def _initilize_store(self):
        try:
            os.makedirs(self.persist_dir, exist_ok=True)

            self.client = chromadb.PersistentClient(path=self.persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Schema Document embeddings for RAG"}
            )

            print(f"[INFO] Chroma initialized. Collection: {self.collection_name}")
            print(f"[INFO] Existing documents: {self.collection.count()}")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Chroma: {e}")
            raise

    def build_documents(self, documents: List[Any]):
        print(f"[INFO] Building Chroma store from {len(documents)} raw documents...")

        embedding_pipline = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = embedding_pipline.chunk_documents(documents)
        embeddings = embedding_pipline.embed_chunks(chunks)

        metadatas = [{"text": chunk.page_content} for chunk in chunks]

        self.add_embeddings(embeddings.astype('float32'), metadatas)
        self.save()

        print(f"[INFO] Chroma vector store built at {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):

        if metadatas is None:
            metadatas = [{} for _ in range(len(embeddings))]

        if len(metadatas) != len(embeddings):
            raise ValueError("Metadata count must match embeddings count")

        print(f"[INFO] Adding {len(embeddings)} vectors to Chroma...")

        ids = []
        documents_text = []
        embeddings_list = []

        for i, (embedding, metadata) in enumerate(zip(embeddings, metadatas)):
            uid = f"vec_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(uid)

            documents_text.append(metadata.get('text', ''))
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text
        )

        print(f"[INFO] Added {len(embeddings)} embeddings to Chroma.")
        print(f"[INFO] Total count: {self.collection.count()}")

    def save(self):
        print(f"[INFO] Chroma auto-persists to: {self.persist_dir}")

    def load(self):
        print(f"[INFO] Reloading Chroma from: {self.persist_dir}")

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

        print(f"[INFO] Loaded Chroma. Count: {self.collection.count()}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):

        if query_embedding.ndim == 1:
            query_embeddings = [query_embedding.tolist()]
        else:
            query_embeddings = query_embedding.tolist()

        chroma_results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
        )

        results = []
        ids = chroma_results.get("ids", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]

        for i in range(len(ids)):
            results.append({
                "index": i,
                "id": ids[i],
                "distance": distances[i] if i < len(distances) else None,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "text": documents[i] if i < len(documents) else "",
            })

        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying Chroma for: '{query_text}'")

        # USE SHARED ENCODER
        encoder = get_sentence_transformer(self.embedding_model)

        # Always encode as a list — ensures consistent shape
        emb = encoder.encode([query_text], show_progress_bar=False)

        query_emb = np.asarray(emb, dtype="float32")[0]

        return self.search(query_emb, top_k=top_k)