# vector_store.py
# -*- coding: utf-8 -*-

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

# -------------------------------
# 1) 문서 리스트 → 임베딩 생성
# -------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    client = get_client()
    # OpenAI v1 style
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")

# -------------------------------
# 2) 벡터 스토어 구축 & 저장
# -------------------------------
def build_vector_store(
    documents: List[Document],
    store_dir: Path = Path("data/vector_store")
) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    texts = [d.text for d in documents]

    print(f"🧠 Embedding {len(texts)} chunks...")
    embeddings = embed_texts(texts)   # (N, D)

    # 벡터 저장
    emb_path = store_dir / "embeddings.npy"
    np.save(emb_path, embeddings)

    # 메타데이터 저장
    meta_path = store_dir / "metadatas.json"
    payload: List[Dict[str, Any]] = []
    for d in documents:
        payload.append({
            "id": d.id,
            "text": d.text,
            "metadata": d.metadata
        })
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Vector store saved at {store_dir}")

# -------------------------------
# 3) 로딩
# -------------------------------
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