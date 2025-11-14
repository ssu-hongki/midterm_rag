# utils.py
# -*- coding: utf-8 -*-

import os
import numpy as np
from typing import Tuple

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