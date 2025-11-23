# rag_chain.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
from openai import OpenAI

from utils import load_env, top_k_similar
from vector_store import load_vector_store

ANSWER_MODEL = "gpt-4.1-mini"  # 과제 요구에 맞게 바꿔도 됨

# Cross-encoder는 필요할 때만 로드 (lazy loading)
_cross_encoder = None

def get_cross_encoder():
    """Cross-encoder 모델을 lazy loading으로 가져옵니다."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            # 한국어 지원이 좋은 모델 사용
            _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except ImportError:
            print("⚠️ sentence-transformers가 설치되지 않았습니다. 재랭킹 기능을 사용할 수 없습니다.")
            return None
    return _cross_encoder

class RAGChain:
    def __init__(self, k: int = 5, use_reranking: bool = True, use_query_expansion: bool = True):
        load_env()
        self.client = OpenAI()
        self.k = k
        self.use_reranking = use_reranking
        self.use_query_expansion = use_query_expansion
        self.embeddings, self.metadatas = load_vector_store()

    def _transform_query(self, query: str) -> str:
        """질문을 더 명확하고 검색하기 좋은 형태로 변환합니다."""
        prompt = f"""다음 질문을 강의계획서 검색에 적합한 명확한 질문으로 변환해주세요.
- 오타나 문법 오류를 수정하세요
- 모호한 표현을 구체적으로 바꾸세요
- 강의계획서에서 찾을 수 있는 정보 형태로 변환하세요
- 원래 의도를 유지하면서 더 명확하게 표현하세요

원본 질문: {query}

변환된 질문:"""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "너는 질문을 명확하게 변환하는 전문가야. 원래 의도를 유지하면서 더 검색하기 좋은 형태로 바꿔줘."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            transformed = resp.choices[0].message.content.strip()
            # 따옴표 제거
            transformed = transformed.strip('"').strip("'")
            return transformed
        except Exception as e:
            print(f"⚠️ 질문 변환 실패: {e}. 원본 질문 사용.")
            return query

    def _expand_query(self, query: str) -> List[str]:
        """질문을 여러 관점에서 확장하여 다중 질문을 생성합니다."""
        prompt = f"""다음 질문을 강의계획서 검색에 도움이 되도록 2-3개의 관련 질문으로 확장해주세요.
각 질문은 서로 다른 관점이나 표현을 사용하되, 원래 질문의 핵심 의도를 유지해야 합니다.

원본 질문: {query}

생성할 질문들 (각 줄에 하나씩, 번호 없이):"""

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "너는 질문 확장 전문가야. 원래 질문의 의도를 유지하면서 다양한 표현으로 질문을 확장해줘."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            expanded_text = resp.choices[0].message.content.strip()
            # 줄바꿈으로 분리하고 정제
            queries = [q.strip().strip('-').strip() for q in expanded_text.split('\n') if q.strip()]
            # 원본 질문도 포함
            queries.insert(0, query)
            # 중복 제거
            seen = set()
            unique_queries = []
            for q in queries:
                if q and q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            return unique_queries[:3]  # 최대 3개
        except Exception as e:
            print(f"⚠️ 질문 확장 실패: {e}. 원본 질문만 사용.")
            return [query]

    def _embed_query(self, query: str):
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        return resp.data[0].embedding

    def _retrieve(self, query: str, transformed_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """벡터 검색으로 후보를 찾고, 재랭킹을 적용합니다."""
        search_query = transformed_query if transformed_query else query
        
        # Query Expansion: 다중 질문으로 검색
        if self.use_query_expansion:
            expanded_queries = self._expand_query(search_query)
        else:
            expanded_queries = [search_query]
        
        # 각 확장된 질문으로 검색하고 결과 통합
        all_candidates = {}
        candidate_k = self.k * 3 if self.use_reranking else self.k * 2
        
        for eq in expanded_queries:
            q_vec = self._embed_query(eq)
            scores, idxs = top_k_similar(self.embeddings, q_vec, k=candidate_k)
            
            for score, idx in zip(scores, idxs):
                item_id = self.metadatas[int(idx)]["id"]
                if item_id not in all_candidates:
                    item = dict(self.metadatas[int(idx)])
                    item["vector_score"] = float(score)
                    item["score"] = float(score)
                    item["matched_queries"] = [eq]
                    all_candidates[item_id] = item
                else:
                    # 여러 질문에서 매칭된 경우 점수 평균
                    all_candidates[item_id]["vector_score"] = (
                        all_candidates[item_id]["vector_score"] + float(score)
                    ) / 2
                    all_candidates[item_id]["score"] = all_candidates[item_id]["vector_score"]
                    if eq not in all_candidates[item_id]["matched_queries"]:
                        all_candidates[item_id]["matched_queries"].append(eq)
        
        candidates = list(all_candidates.values())
        # 점수로 정렬
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # 2단계: 재랭킹 적용 (옵션)
        if self.use_reranking and len(candidates) > self.k:
            candidates = self._rerank(search_query, candidates[:candidate_k])
        
        # 상위 k개만 반환
        return candidates[:self.k]

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cross-encoder를 사용하여 후보들을 재랭킹합니다."""
        cross_encoder = get_cross_encoder()
        if cross_encoder is None:
            # 재랭킹 모델이 없으면 벡터 점수로 정렬
            return sorted(candidates, key=lambda x: x["vector_score"], reverse=True)

        # 쿼리-문서 쌍 생성
        pairs = [[query, candidate["text"]] for candidate in candidates]

        # Cross-encoder로 점수 계산
        try:
            rerank_scores = cross_encoder.predict(pairs)
        except Exception as e:
            print(f"⚠️ 재랭킹 중 오류 발생: {e}. 벡터 점수로 정렬합니다.")
            return sorted(candidates, key=lambda x: x["vector_score"], reverse=True)

        # 재랭킹 점수를 추가하고 정렬
        for candidate, rerank_score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(rerank_score)
            # 벡터 점수와 재랭킹 점수를 결합 (가중 평균)
            # 재랭킹 점수를 더 높게 가중치 부여
            candidate["score"] = 0.3 * candidate["vector_score"] + 0.7 * float(rerank_score)

        # 재랭킹 점수로 정렬
        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return reranked

    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        context_texts = []
        for c in contexts:
            src = c["metadata"].get("강좌명") or c["metadata"].get("과목코드") or c.get("id", "")
            header = f"[출처: {src}]"
            
            # text와 함께 metadata의 중요한 정보도 포함
            text_content = c["text"]
            meta = c.get("metadata", {})
            
            # metadata에서 추가 정보 추출 (text에 없을 수 있는 정보)
            additional_info = []
            if meta.get("교수실") and meta["교수실"] != "없음" and "교수실" not in text_content:
                additional_info.append(f"교수실 연락처: {meta['교수실']}")
            if meta.get("연락처") and meta["연락처"] != "없음" and meta.get("연락처") != meta.get("교수실") and "연락처" not in text_content:
                additional_info.append(f"연락처: {meta['연락처']}")
            if meta.get("이메일") and meta["이메일"] != "없음" and "이메일" not in text_content:
                additional_info.append(f"이메일: {meta['이메일']}")
            
            if additional_info:
                text_content += "\n" + "\n".join(additional_info)
            
            context_texts.append(header + "\n" + text_content)

        context_block = "\n\n---\n\n".join(context_texts)

        system_msg = (
            "너는 숭실대학교 강의계획서 RAG 챗봇이야.\n"
            "아래 제공된 강의계획서 청크 내용만을 근거로 답변하고, "
            "모르는 내용은 아는 척 하지 말고 모른다고 말해."
        )

        user_msg = (
            f"다음은 강의계획서에서 추출한 관련 내용이야:\n\n"
            f"{context_block}\n\n"
            f"위 내용을 참고해서 아래 질문에 한국어로 자세히 답변해줘.\n"
            f"질문: {query}"
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def ask(self, query: str) -> Dict[str, Any]:
        # 1단계: 질문 변환 (명확하게 만들기)
        transformed_query = self._transform_query(query) if self.use_query_expansion else query
        
        # 2단계: 검색
        contexts = self._retrieve(query, transformed_query=transformed_query)
        
        # 3단계: 프롬프트 생성 (원본 질문 사용)
        messages = self._build_prompt(query, contexts)

        resp = self.client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=messages,
        )
        answer = resp.choices[0].message.content

        return {
            "answer": answer,
            "contexts": contexts,
            "original_query": query,
            "transformed_query": transformed_query if transformed_query != query else None,
        }