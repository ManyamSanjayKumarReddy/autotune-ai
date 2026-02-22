from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from rag_pipeline.encoder import get_sentence_transformer

class EmbeddingPipeline:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        # Use shared encoder (cached)
        self.model = get_sentence_transformer(model_name)
        print(f"[INFO] EmbeddingPipeline initialized with model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            # removed empty-string separator to avoid single-character chunks
            separators=["\n\n", "\n", " "]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings