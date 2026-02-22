from sentence_transformers import SentenceTransformer
from typing import Optional
import os

# module-level cache so the model is instantiated only once per process
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_SHARED_ENCODER: Optional[SentenceTransformer] = None
_SHARED_MODEL_NAME: Optional[str] = None

def get_sentence_transformer(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    """
    Return a cached SentenceTransformer instance, forced onto CPU to avoid
    PyTorch meta-tensor initialization errors on newer torch versions.
    """

    global _SHARED_ENCODER, _SHARED_MODEL_NAME

    if _SHARED_ENCODER is None:
        print(f"[INFO] Loading SentenceTransformer model on CPU: {model_name}")
        _SHARED_ENCODER = SentenceTransformer(model_name, device="cpu")
        _SHARED_MODEL_NAME = model_name
    elif _SHARED_MODEL_NAME != model_name:
        print(f"[WARN] Requested '{model_name}' but cached '{_SHARED_MODEL_NAME}'. Reusing cached.")

    return _SHARED_ENCODER