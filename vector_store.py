from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np

from openai import OpenAI

from data_loader import Document
from utils import load_env

EMBED_MODEL = "text-embedding-3-small"

def get_client() -> OpenAI:
    load_env()
    return OpenAI()

def metadata_to_text(metadata: Dict[str, Any]) -> str:
    parts = []
    
    def extract_strings(obj, prefix=""):
        if isinstance(obj, str):
            parts.append(obj)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                extract_strings(value, f"{prefix}{key}: ")
        elif isinstance(obj, list):
            for item in obj:
                extract_strings(item, prefix)
        elif obj is not None:
            parts.append(str(obj))
    
    extract_strings(metadata)
    return " ".join(parts)

def embed_texts(texts: List[str]) -> np.ndarray:
    client = get_client()
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")

def build_vector_store(
    documents: List[Document],
    store_dir: Path = Path("data/vector_store")
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    
    texts = [metadata_to_text(d.metadata) for d in documents]

    embeddings = embed_texts(texts)

    emb_path = store_dir / "embeddings.npy"
    np.save(emb_path, embeddings)

    meta_path = store_dir / "metadatas.json"
    payload: List[Dict[str, Any]] = []
    for d in documents:
        payload.append({
            "chunk_id": d.id,
            "source_pdf": d.metadata.get("source_pdf", ""),
            "metadata": d.metadata
        })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_vector_store(
    store_dir: Path = Path("data/vector_store")
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    emb_path = store_dir / "embeddings.npy"
    meta_path = store_dir / "metadatas.json"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("벡터 스토어 파일이 없습니다. 먼저 build_vector_store를 실행하세요.")

    embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    return embeddings, metadatas