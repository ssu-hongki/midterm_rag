# rag_chain.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
from openai import OpenAI

from utils import load_env, top_k_similar, initialize_bm25, hybrid_search
from vector_store import load_vector_store

ANSWER_MODEL = "gpt-4o-mini"  # 과제 요구에 맞게 바꿔도 됨

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
    def __init__(self, k: int = 5, use_reranking: bool = True, use_query_expansion: bool = True, use_hybrid_search: bool = True):
        load_env()
        self.client = OpenAI()
        self.k = k
        self.use_reranking = use_reranking
        self.use_query_expansion = use_query_expansion
        self.use_hybrid_search = use_hybrid_search
        self.embeddings, self.metadatas = load_vector_store()
        
        # BM25 초기화 (하이브리드 검색용)
        if self.use_hybrid_search:
            documents = [meta.get("text", "") for meta in self.metadatas]
            if initialize_bm25(documents):
                print("✅ BM25 인덱스 초기화 완료")
            else:
                print("⚠️ BM25 초기화 실패. 벡터 검색만 사용합니다.")
                self.use_hybrid_search = False

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

    def _extract_metadata_filters(self, query: str) -> Dict[str, Any]:
        """질문에서 metadata 필터 조건을 추출합니다."""
        filters = {}
        
        # 학과/학부 필터 추출
        학과_키워드 = ["학부", "학과", "대상"]
        학과_목록 = ["ai융합", "ai융합학부", "컴퓨터", "컴퓨터학부", "소프트웨어", "전자정보", "전자정보공학부"]
        
        query_lower = query.lower()
        for 학과 in 학과_목록:
            if 학과 in query_lower:
                filters["수강대상학과"] = 학과
                break
        
        # 학년 필터 추출
        import re
        학년_패턴 = r"(\d)학년"
        match = re.search(학년_패턴, query)
        if match:
            학년 = match.group(1)
            filters["학년"] = f"{학년}학년"
        
        # 강좌명 필터 추출 (간단한 키워드 매칭)
        강좌명_키워드 = ["자연언어처리", "nlp", "데이터베이스", "db", "프로그래밍", "프로그래밍및실습"]
        for 키워드 in 강좌명_키워드:
            if 키워드 in query_lower:
                if "nlp" in query_lower or "자연언어처리" in query_lower:
                    filters["강좌명_키워드"] = "자연언어처리"
                elif "db" in query_lower or "데이터베이스" in query_lower:
                    filters["강좌명_키워드"] = "데이터베이스"
                elif "프로그래밍" in query_lower:
                    filters["강좌명_키워드"] = "프로그래밍"
                break
        
        # LLM을 사용한 더 정확한 필터 추출
        if not filters:  # 간단한 추출로 필터를 못 찾았을 때만 LLM 사용
            try:
                prompt = f"""다음 질문에서 강의계획서 검색에 필요한 필터 조건을 추출해주세요.
질문: {query}

다음 JSON 형식으로 답변해주세요 (해당하는 것만):
{{
  "수강대상학과": "학과/학부명 (예: AI융합학부, 컴퓨터학부)",
  "학년": "학년 (예: 1학년, 2학년, 3학년, 4학년)",
  "강좌명": "강좌명 (예: 자연언어처리, 데이터베이스)",
  "담당교수": "교수명",
  "과목코드": "과목코드"
}}

해당하는 정보가 없으면 null로 표시하세요. JSON만 답변하세요."""

                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "너는 질문에서 필터 조건을 추출하는 전문가야. JSON 형식으로만 답변해줘."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=200
                )
                import json
                extracted = json.loads(resp.choices[0].message.content.strip())
                filters.update({k: v for k, v in extracted.items() if v and v != "null"})
            except Exception as e:
                print(f"⚠️ 필터 추출 실패: {e}")
        
        return filters

    def _filter_by_metadata(self, filters: Dict[str, Any]) -> List[int]:
        """metadata 필터 조건에 맞는 청크 인덱스를 반환합니다."""
        if not filters:
            return list(range(len(self.metadatas)))  # 필터가 없으면 전체
        
        filtered_indices = []
        
        for idx, meta_item in enumerate(self.metadatas):
            metadata = meta_item.get("metadata", {})
            match = True
            
            # 수강대상학과 필터
            if "수강대상학과" in filters:
                수강대상 = metadata.get("수강대상학과", "").lower()
                필터_학과 = filters["수강대상학과"].lower()
                # "ai융합" -> "ai융합학부" 매칭
                if "ai융합" in 필터_학과:
                    if "ai융합" not in 수강대상:
                        match = False
                elif "컴퓨터" in 필터_학과:
                    if "컴퓨터" not in 수강대상:
                        match = False
                else:
                    if 필터_학과 not in 수강대상:
                        match = False
            
            # 학년 필터
            if match and "학년" in filters:
                수강대상 = metadata.get("수강대상학과", "").lower()
                필터_학년 = filters["학년"].lower()
                # 수강대상에 학년 정보가 포함되어 있는지 확인
                # "3학년" 또는 "3" 모두 매칭
                학년_숫자 = 필터_학년.replace("학년", "").strip()
                if 학년_숫자 not in 수강대상 and 필터_학년 not in 수강대상:
                    match = False
            
            # 강좌명 필터
            if match and "강좌명_키워드" in filters:
                강좌명 = metadata.get("강좌명", "").lower()
                필터_키워드 = filters["강좌명_키워드"].lower()
                if 필터_키워드 not in 강좌명:
                    match = False
            
            if match and "강좌명" in filters:
                강좌명 = metadata.get("강좌명", "").lower()
                필터_강좌명 = filters["강좌명"].lower()
                if 필터_강좌명 not in 강좌명:
                    match = False
            
            # 담당교수 필터
            if match and "담당교수" in filters:
                교수 = metadata.get("담당교수", "").lower()
                필터_교수 = filters["담당교수"].lower()
                if 필터_교수 not in 교수:
                    match = False
            
            # 과목코드 필터
            if match and "과목코드" in filters:
                코드 = metadata.get("과목코드", "")
                필터_코드 = filters["과목코드"]
                if 필터_코드 not in str(코드):
                    match = False
            
            if match:
                filtered_indices.append(idx)
        
        return filtered_indices

    def _embed_query(self, query: str):
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        return resp.data[0].embedding

    def _retrieve(self, query: str, transformed_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """하이브리드 검색으로 Top-20 후보를 찾고, 재랭킹으로 Top-N을 선택합니다."""
        search_query = transformed_query if transformed_query else query
        
        # 1단계: Metadata 필터링
        filters = self._extract_metadata_filters(query)
        filtered_indices = self._filter_by_metadata(filters)
        
        if not filtered_indices:
            # 필터에 맞는 청크가 없으면 빈 리스트 반환
            return []
        
        # 필터링된 청크의 임베딩만 사용
        filtered_embeddings = self.embeddings[filtered_indices]
        filtered_metadatas = [self.metadatas[idx] for idx in filtered_indices]
        index_mapping = {i: original_idx for i, original_idx in enumerate(filtered_indices)}
        
        # Query Expansion: 다중 질문으로 검색
        if self.use_query_expansion:
            expanded_queries = self._expand_query(search_query)
        else:
            expanded_queries = [search_query]
        
        # 각 확장된 질문으로 검색하고 결과 통합
        all_candidates = {}
        # 재순위를 사용할 경우 Top-20 후보, 아니면 k*2
        candidate_k = 20 if self.use_reranking else self.k * 2
        
        for eq in expanded_queries:
            q_vec = self._embed_query(eq)
            
            # 하이브리드 검색 (BM25 + Vector)
            # 필터링이 적용된 경우 BM25 인덱스 불일치 문제로 벡터 검색만 사용
            if self.use_hybrid_search and len(filtered_indices) == len(self.metadatas):
                # 필터링이 없는 경우(전체 문서)만 하이브리드 검색 사용
                hybrid_results = hybrid_search(
                    query=eq,
                    query_embedding=q_vec,
                    embeddings=self.embeddings,
                    metadatas=self.metadatas,
                    k=min(candidate_k, len(self.embeddings)),
                    alpha=0.5  # 벡터와 BM25 점수의 가중치 (0.5 = 동일 비중)
                )
                
                for item in hybrid_results:
                    item_id = item["id"]
                    if item_id not in all_candidates:
                        item["matched_queries"] = [eq]
                        all_candidates[item_id] = item
                    else:
                        # 여러 질문에서 매칭된 경우 점수 평균
                        all_candidates[item_id]["vector_score"] = (
                            all_candidates[item_id]["vector_score"] + item["vector_score"]
                        ) / 2
                        all_candidates[item_id]["bm25_score"] = (
                            all_candidates[item_id]["bm25_score"] + item["bm25_score"]
                        ) / 2
                        all_candidates[item_id]["hybrid_score"] = (
                            all_candidates[item_id]["hybrid_score"] + item["hybrid_score"]
                        ) / 2
                        all_candidates[item_id]["score"] = all_candidates[item_id]["hybrid_score"]
                        if eq not in all_candidates[item_id]["matched_queries"]:
                            all_candidates[item_id]["matched_queries"].append(eq)
            else:
                # 벡터 검색만 사용 (필터링이 적용된 경우 또는 하이브리드 비활성화)
                scores, local_idxs = top_k_similar(filtered_embeddings, q_vec, k=min(candidate_k, len(filtered_embeddings)))
                
                for score, local_idx in zip(scores, local_idxs):
                    original_idx = index_mapping[int(local_idx)]
                    item_id = self.metadatas[original_idx]["id"]
                    
                    if item_id not in all_candidates:
                        item = dict(self.metadatas[original_idx])
                        item["vector_score"] = float(score)
                        item["bm25_score"] = 0.0
                        item["hybrid_score"] = float(score)
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
        
        # 2단계: Cross-Encoder 재랭킹 적용
        # Top-20 후보를 Cross-Encoder로 재평가하여 Top-N(3~5) 선택
        if self.use_reranking and len(candidates) >= self.k:
            # 후보가 k개 이상이면 재랭킹 적용 (최대 candidate_k개까지)
            candidates = self._rerank(search_query, candidates[:min(candidate_k, len(candidates))])
        
        # 상위 k개만 반환 (최종 Top-N)
        return candidates[:self.k]

    def _rerank_with_llm(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM API를 사용하여 후보들을 재랭킹합니다.
        
        Cross-Encoder 대신 GPT 모델로 관련성을 평가합니다.
        """
        if len(candidates) <= self.k:
            return candidates

        # 배치로 처리 (API 호출 최소화)
        batch_size = 5
        all_scored = []

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            
            # 후보 문서들을 텍스트로 구성
            docs_text = ""
            for idx, candidate in enumerate(batch):
                docs_text += f"\n[문서 {idx+1}]\n{candidate['text'][:500]}...\n"  # 토큰 제한을 위해 500자로 제한

            prompt = f"""다음 질문과 각 문서의 관련성을 0~10 점수로 평가해주세요.

질문: {query}

문서들:
{docs_text}

각 문서의 관련성 점수를 JSON 배열로 답변해주세요:
[점수1, 점수2, 점수3, ...]

예시: [8.5, 3.2, 9.1]"""

            try:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",  # 비용 절감을 위해 mini 사용
                    messages=[
                        {"role": "system", "content": "너는 문서 관련성 평가 전문가야. 점수만 JSON 배열로 답변해."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=100
                )
                
                import json
                scores_text = resp.choices[0].message.content.strip()
                # JSON 파싱 시도
                scores = json.loads(scores_text)
                
                # 점수를 candidates에 추가
                for candidate, score in zip(batch, scores):
                    candidate["rerank_score"] = float(score) / 10.0  # 0~1로 정규화
                    hybrid_score = candidate.get("hybrid_score", candidate.get("vector_score", 0))
                    candidate["score"] = 0.3 * hybrid_score + 0.7 * candidate["rerank_score"]
                    all_scored.append(candidate)
                    
            except Exception as e:
                print(f"⚠️ LLM 재랭킹 실패: {e}. 하이브리드 점수 사용")
                # 재랭킹 실패시 원본 점수 유지
                for candidate in batch:
                    candidate["rerank_score"] = candidate.get("hybrid_score", candidate.get("vector_score", 0))
                    candidate["score"] = candidate["rerank_score"]
                    all_scored.append(candidate)

        # 점수로 정렬
        all_scored.sort(key=lambda x: x["score"], reverse=True)
        return all_scored

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """재랭킹 방법을 선택합니다."""
        # LLM API 기반 재랭킹 사용
        return self._rerank_with_llm(query, candidates)
        
        # 또는 원본 Cross-Encoder 사용하려면:
        # cross_encoder = get_cross_encoder()
        # if cross_encoder is None:
        #     return self._rerank_with_llm(query, candidates)
        # ...existing cross-encoder code...

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
        
        # 2단계: Metadata 필터 추출
        filters = self._extract_metadata_filters(query)
        
        # 3단계: 검색
        contexts = self._retrieve(query, transformed_query=transformed_query)
        
        # 4단계: 프롬프트 생성 (원본 질문 사용)
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
            "metadata_filters": filters if filters else None,
        }

    def retrieve_contexts(self, query: str, top_k: int = 5):
        # Vector 검색
        vector_results = self.vector_store.similarity_search_with_score(query, k=top_k*2)
        
        # 메타데이터 필터링 추가 (교재 관련 청크 우선)
        for doc, score in vector_results:
            metadata = doc.metadata
            content = doc.page_content.lower()
            
            # "교재", "참고도서" 등의 키워드가 있으면 점수 부스팅
            if any(keyword in content for keyword in ["교재", "참고도서", "참고문헌", "textbook"]):
                score *= 1.2  # 점수 상향 조정
            
            # ...existing code...

def get_relevant_contexts(
    query: str,
    vectorstore,
    bm25_index,
    reranker,
    k: int = 20,
    final_k: int = 5
) -> List[Dict]:
    """하이브리드 검색 + Reranking으로 관련 문서 반환"""
    
    # 쿼리 확장: 유사 키워드 추가
    query_expansion = {
        #"주요교재": ["교재", "주교재", "참고서적", "교과서", "textbook", "참고도서"],
        #"참고교재": ["부교재", "참고서적", "추천도서", "reference book"],
        "교수": ["담당교수", "교수님", "강의자", "instructor"],
        "학점": ["성적", "평가", "점수", "grade"]
    }
    
    expanded_query = query
    for key, synonyms in query_expansion.items():
        if key in query:
            expanded_query = query + " " + " ".join(synonyms)
            break
    
    # Vector Search
    vector_results = vectorstore.similarity_search_with_score(expanded_query, k=k)
    
    # BM25 Search (확장된 쿼리 사용)
    bm25_scores = bm25_index["index"].get_scores(
        bm25_index["tokenizer"](expanded_query)
    )