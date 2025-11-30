import os
import numpy as np
from typing import Tuple, List, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def load_env():
    if load_dotenv is not None:
        load_dotenv()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a_norm, b_norm)

def top_k_similar(
    matrix: np.ndarray,
    query_vec: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    sims = cosine_sim(matrix, query_vec)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

_bm25_instance = None
_bm25_corpus = None

def initialize_bm25(documents: List[str]):
    global _bm25_instance, _bm25_corpus
    try:
        from rank_bm25 import BM25Okapi
        
        filtered_docs = []
        for doc in documents:
            if doc and doc.strip():
                filtered_docs.append(doc)
            else:
                filtered_docs.append("empty_document")
        
        tokenized_corpus = [doc.split() for doc in filtered_docs]
        tokenized_corpus = [tokens if tokens else ["empty"] for tokens in tokenized_corpus]
        
        _bm25_instance = BM25Okapi(tokenized_corpus)
        _bm25_corpus = filtered_docs
        return True
    except ImportError:
        return False
    except Exception:
        return False

def bm25_search(query: str, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    global _bm25_instance, _bm25_corpus
    
    if _bm25_instance is None:
        raise RuntimeError("BM25가 초기화되지 않았습니다. initialize_bm25()를 먼저 호출하세요.")
    
    tokenized_query = query.split()
    scores = _bm25_instance.get_scores(tokenized_query)
    
    top_indices = np.argsort(-scores)[:k]
    top_scores = scores[top_indices]
    
    return top_scores, top_indices

def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    metadatas: List[Dict[str, Any]],
    k: int = 20,
    alpha: float = 0.5
) -> List[Dict[str, Any]]:
    vector_scores, vector_indices = top_k_similar(embeddings, query_embedding, k=k)
    
    try:
        bm25_scores, bm25_indices = bm25_search(query, k=k)
    except Exception:
        bm25_scores = np.array([])
        bm25_indices = np.array([])
    
    vector_score_dict = {}
    if len(vector_scores) > 0 and vector_scores.max() > 0:
        for idx, score in zip(vector_indices, vector_scores):
            if idx < len(metadatas):
                vector_score_dict[int(idx)] = float(score / vector_scores.max())
    
    bm25_score_dict = {}
    if len(bm25_scores) > 0 and bm25_scores.max() > 0:
        for idx, score in zip(bm25_indices, bm25_scores):
            if idx < len(metadatas):
                bm25_score_dict[int(idx)] = float(score / bm25_scores.max())
    
    all_indices = set(vector_score_dict.keys()) | set(bm25_score_dict.keys())
    results = []
    
    for idx in all_indices:
        v_score = vector_score_dict.get(idx, 0.0)
        b_score = bm25_score_dict.get(idx, 0.0)
        hybrid_score = alpha * v_score + (1 - alpha) * b_score
        
        results.append({
            "metadata": metadatas[idx],
            "hybrid_score": hybrid_score,
            "vector_score": v_score,
            "bm25_score": b_score,
            "filtered_idx": idx
        })
    
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    return results[:k]