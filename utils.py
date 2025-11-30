# utils.py
# -*- coding: utf-8 -*-

import os
import numpy as np
from typing import Tuple, List, Dict, Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def load_env():
    """ .env 파일에서 OPENAI_API_KEY 등을 읽어옴 """
    if load_dotenv is not None:
        load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠ OPENAI_API_KEY가 환경변수에 없습니다. .env를 확인하세요.")

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N, D), b: (D,) 일 때 코사인 유사도 (N,)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a_norm, b_norm)

def top_k_similar(
    matrix: np.ndarray,
    query_vec: np.ndarray,
    k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    matrix: (N, D) 전체 임베딩
    query_vec: (D,) 쿼리 임베딩
    return: (top_k_scores, top_k_indices)
    """
    sims = cosine_sim(matrix, query_vec)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx


# BM25 검색 관련 함수들
_bm25_instance = None
_bm25_corpus = None

def initialize_bm25(documents: List[str]):
    """BM25 인덱스를 초기화합니다."""
    global _bm25_instance, _bm25_corpus
    try:
        from rank_bm25 import BM25Okapi
        
        # 한국어 토크나이저 (간단한 공백 기반)
        tokenized_corpus = [doc.split() for doc in documents]
        _bm25_instance = BM25Okapi(tokenized_corpus)
        _bm25_corpus = documents
        return True
    except ImportError:
        print("⚠️ rank-bm25가 설치되지 않았습니다. BM25 검색을 사용할 수 없습니다.")
        return False
    except Exception as e:
        print(f"⚠️ BM25 초기화 중 오류 발생: {e}")
        return False

def bm25_search(query: str, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    BM25로 문서를 검색합니다.
    
    Args:
        query: 검색 쿼리
        k: 반환할 상위 문서 개수
    
    Returns:
        (scores, indices): BM25 점수와 문서 인덱스
    """
    global _bm25_instance, _bm25_corpus
    
    if _bm25_instance is None:
        raise RuntimeError("BM25가 초기화되지 않았습니다. initialize_bm25()를 먼저 호출하세요.")
    
    tokenized_query = query.split()
    scores = _bm25_instance.get_scores(tokenized_query)
    
    # 상위 k개 인덱스 추출
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
    """
    하이브리드 검색: BM25와 벡터 검색을 결합합니다.
    
    Args:
        query: 검색 쿼리
        query_embedding: 쿼리 임베딩 벡터
        embeddings: 문서 임베딩 행렬
        metadatas: 문서 메타데이터 리스트
        k: 반환할 상위 문서 개수
        alpha: 벡터 검색 가중치 (1-alpha는 BM25 가중치)
    
    Returns:
        하이브리드 점수로 정렬된 문서 리스트
    """
    # 1. 벡터 검색
    vector_scores, vector_indices = top_k_similar(embeddings, query_embedding, k=k)
    
    # 2. BM25 검색
    try:
        bm25_scores, bm25_indices = bm25_search(query, k=k)
    except Exception as e:
        print(f"⚠️ BM25 검색 실패: {e}. 벡터 검색 결과만 사용합니다.")
        # BM25 실패 시 벡터 검색 결과만 반환
        results = []
        for score, idx in zip(vector_scores, vector_indices):
            item = dict(metadatas[int(idx)])
            item["vector_score"] = float(score)
            item["bm25_score"] = 0.0
            item["hybrid_score"] = float(score)
            item["score"] = float(score)
            results.append(item)
        return results
    
    # 3. 점수 정규화
    # 벡터 검색 점수 정규화 (0~1 범위로)
    vector_score_dict = {}
    if len(vector_scores) > 0 and vector_scores.max() > 0:
        normalized_vector_scores = vector_scores / vector_scores.max()
        for idx, score in zip(vector_indices, normalized_vector_scores):
            vector_score_dict[int(idx)] = float(score)
    
    # BM25 점수 정규화 (0~1 범위로)
    bm25_score_dict = {}
    if len(bm25_scores) > 0 and bm25_scores.max() > 0:
        normalized_bm25_scores = bm25_scores / bm25_scores.max()
        for idx, score in zip(bm25_indices, normalized_bm25_scores):
            bm25_score_dict[int(idx)] = float(score)
    
    # 4. 하이브리드 점수 계산 및 결합
    all_indices = set(vector_score_dict.keys()) | set(bm25_score_dict.keys())
    results = []
    
    for idx in all_indices:
        v_score = vector_score_dict.get(idx, 0.0)
        b_score = bm25_score_dict.get(idx, 0.0)
        hybrid_score = alpha * v_score + (1 - alpha) * b_score
        
        item = dict(metadatas[idx])
        item["vector_score"] = v_score
        item["bm25_score"] = b_score
        item["hybrid_score"] = hybrid_score
        item["score"] = hybrid_score
        results.append(item)
    
    # 하이브리드 점수로 정렬
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    return results[:k]