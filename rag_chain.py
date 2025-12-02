# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional
from openai import OpenAI
import numpy as np

from utils import load_env, top_k_similar, initialize_bm25, hybrid_search
from vector_store import load_vector_store

ANSWER_MODEL = "gpt-4o-mini"

_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        except ImportError:
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
        
        if self.use_hybrid_search:
            documents = [meta.get("text", "") for meta in self.metadatas]
            initialize_bm25(documents)
            self.use_hybrid_search = False

    def _transform_query(self, query: str) -> str:
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
            transformed = transformed.strip('"').strip("'")
            return transformed
        except Exception:
            return query

    def _expand_query(self, query: str) -> List[str]:
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
            queries = [q.strip().strip('-').strip() for q in expanded_text.split('\n') if q.strip()]
            queries.insert(0, query)
            seen = set()
            unique_queries = []
            for q in queries:
                if q and q not in seen:
                    seen.add(q)
                    unique_queries.append(q)
            return unique_queries[:3]
        except Exception as e:
            print(f"질문 확장 실패: {e}. 원본 질문만 사용.")
            return [query]

    def _extract_metadata_filters(self, query: str) -> Dict[str, Any]:
        filters = {}
        
        import re
        query_lower = query.lower()

        grade_pattern = r"(\d)학년"
        match = re.search(grade_pattern, query)
        if match:
            grade = match.group(1)
            filters["학년"] = f"{grade}학년"
        
        course_keyword_map = {
            "자연언어처리": ["자연언어처리", "nlp"],
            "데이터베이스": ["데이터베이스", "db"],
            "프로그래밍": ["프로그래밍"],
            "컴퓨터비전": ["컴퓨터비전", "computer vision", "cv"],
            "머신러닝": ["머신러닝", "machine learning", "ml"],
            "딥러닝": ["딥러닝", "deep learning", "dl"],
        }
        
        for course_name, keywords in course_keyword_map.items():
            if any(keyword in query_lower for keyword in keywords):
                filters["강좌명_키워드"] = course_name
                break
        
        if not filters:
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
                print(f"필터 추출 실패: {e}")
        
        return filters

    def _filter_by_metadata(self, filters: Dict[str, Any]) -> List[int]:
        if not filters:
            return list(range(len(self.metadatas)))
        
        filtered_indices = []
        
        for idx, meta_item in enumerate(self.metadatas):
            metadata = meta_item.get("metadata", {})
            match = True
            
            if match and "학년" in filters:
                target_dept = metadata.get("수강대상학과", "").lower()
                filter_grade = filters["학년"].lower()
                grade_num = filter_grade.replace("학년", "").strip()
                if grade_num not in target_dept and filter_grade not in target_dept:
                    match = False
            
            if match and "강좌명_키워드" in filters:
                course_name = metadata.get("강좌명", "").lower()
                filter_keyword = filters["강좌명_키워드"].lower()
                if filter_keyword not in course_name:
                    match = False
            
            if match and "강좌명" in filters:
                course_name = metadata.get("강좌명", "").lower()
                filter_course = filters["강좌명"].lower()
                if filter_course not in course_name:
                    match = False
            
            if match and "담당교수" in filters:
                professor = metadata.get("담당교수", "").lower()
                filter_prof = filters["담당교수"].lower()
                if filter_prof not in professor:
                    match = False
            
            if match and "과목코드" in filters:
                code = metadata.get("과목코드", "")
                filter_code = filters["과목코드"]
                if filter_code not in str(code):
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

    def _retrieve(
        self,
        query: str,
        transformed_query: Optional[str] = None
    ):
        if self.use_query_expansion:
            expanded_queries = self._expand_query(transformed_query or query)
            print(f"\n쿼리 확장: {len(expanded_queries)}개")
            for i, eq in enumerate(expanded_queries, 1):
                print(f"  {i}. {eq}")
        else:
            expanded_queries = [query]
        
        filters = self._extract_metadata_filters(query)
        
        if filters:
            valid_indices = self._filter_by_metadata(filters)
            print(f"필터링 결과: {len(valid_indices)}/{len(self.metadatas)}개 문서")
        else:
            valid_indices = list(range(len(self.metadatas)))
            print(f"필터 없음: 전체 {len(self.metadatas)}개 문서 검색")
        
        if not valid_indices:
            print("필터 조건에 맞는 문서가 없습니다!")
            return []
        
        filtered_embeddings = self.embeddings[valid_indices]
        filtered_metadatas = [self.metadatas[i] for i in valid_indices]
        
        all_candidates = {}
        candidate_k = 20
        
        for i, eq in enumerate(expanded_queries, 1):
            eq_emb = self._embed_query(eq)
            
            if self.use_hybrid_search:
                eq_results = hybrid_search(
                    query=eq,
                    query_embedding=np.array(eq_emb),
                    embeddings=filtered_embeddings,
                    metadatas=filtered_metadatas,
                    k=candidate_k,
                    alpha=0.5
                )
                print(f"  쿼리 {i} 하이브리드 검색: {len(eq_results)}개 발견")
            else:
                from utils import top_k_similar
                scores, indices = top_k_similar(
                    filtered_embeddings,
                    np.array(eq_emb),
                    k=min(candidate_k, len(filtered_embeddings))
                )
                eq_results = []
                for score, idx in zip(scores, indices):
                    eq_results.append({
                        "metadata": filtered_metadatas[idx],
                        "score": float(score),
                        "filtered_idx": idx
                    })
                print(f"  쿼리 {i} 벡터 검색: {len(eq_results)}개 발견")
        
            before_merge = len(all_candidates)
            for item in eq_results:
                if "filtered_idx" in item:
                    filtered_idx = item["filtered_idx"]
                else:
                    item_meta = item.get("metadata", item)
                    try:
                        chunk_id = item_meta.get("chunk_id")
                        filtered_idx = next(
                            idx for idx, meta in enumerate(filtered_metadatas)
                            if meta.get("chunk_id") == chunk_id
                        )
                    except (StopIteration, KeyError, TypeError):
                        print(f"    매칭 실패: {item_meta.get('chunk_id', 'unknown')}")
                        continue
                
                original_idx = valid_indices[filtered_idx]
                
                if original_idx not in all_candidates:
                    all_candidates[original_idx] = item
                else:
                    current_score = item.get("hybrid_score", item.get("score", 0))
                    existing_score = all_candidates[original_idx].get("hybrid_score", 
                                    all_candidates[original_idx].get("score", 0))
                    if current_score > existing_score:
                        all_candidates[original_idx] = item
            
            after_merge = len(all_candidates)
            print(f"    → 병합 후: {after_merge}개 (새로 추가: {after_merge - before_merge}개)")
        
        candidates = list(all_candidates.values())
        candidates.sort(key=lambda x: x.get("hybrid_score", x.get("score", 0)), reverse=True)
        
        if self.use_reranking and len(candidates) >= self.k:
            print(f"재순위 적용 중... ({len(candidates)}개 → 상위 {min(candidate_k, len(candidates))}개 대상)")
            candidates = self._rerank(query, candidates[:min(candidate_k, len(candidates))])
            print(f"재순위 완료: {len(candidates)}개")
        
        final = candidates[:self.k]
        return final

    def _rerank_with_cross_encoder(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cross_encoder = get_cross_encoder()
        if cross_encoder is None:
            print("Cross-Encoder를 사용할 수 없습니다. 원본 점수 사용.")
            return candidates
        
        if len(candidates) <= self.k:
            return candidates
        
        texts = []
        for candidate in candidates:
            metadata = candidate.get("metadata", {})
            
            if 'metadata' in metadata and isinstance(metadata['metadata'], dict):
                inner_metadata = metadata['metadata']
            else:
                inner_metadata = metadata
            
            text_parts = []
            
            course_name = inner_metadata.get('강좌명', '')
            if course_name:
                text_parts.append(f"강좌명: {course_name}")
                text_parts.append(f"과목: {course_name}")
                text_parts.append(f"교과목: {course_name}")
            
            important_fields = [
                '담당교수', '교과목 개요', '교육목표', '강의개요', '강의내용',
                '주요교재', '참고교재(대표)', '수업방법', '평가항목', 
                '주차별 강의개요', '학습준비사항', '수강학생 유의 및 참고사항',
                '수강대상학과', '학점', '주당시간', '강좌형식',
                '필수 선수과목', '권장 선수과목'
            ]
            
            for key in important_fields:
                if key in inner_metadata and inner_metadata[key]:
                    value = inner_metadata[key]
                    if isinstance(value, list):
                        text_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
                    elif isinstance(value, dict):
                        text_parts.append(f"{key}: {str(value)}")
                    else:
                        text_parts.append(f"{key}: {value}")
            
            for key, value in inner_metadata.items():
                if key not in important_fields and key not in ['chunk_id', 'source_pdf', '강좌명'] and value:
                    if not isinstance(value, (list, dict)):
                        text_parts.append(f"{key}: {value}")
            
            text_content = "\n".join(text_parts)
            
            if len(text_content) < 100:
                text_content = str(candidate)[:2000]
            
            texts.append(text_content)
        
        pairs = [(query, text) for text in texts]
        
        try:
            scores = cross_encoder.predict(pairs)
            
            min_score = float(min(scores))
            max_score = float(max(scores))
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            print(f"\nCross-Encoder 점수 범위: {min_score:.3f} ~ {max_score:.3f}")
            
            for candidate, score in zip(candidates, scores):
                normalized_score = (float(score) - min_score) / score_range
                candidate["rerank_score"] = normalized_score
                
                hybrid_score = candidate.get("hybrid_score", candidate.get("vector_score", 0))
                
                candidate["score"] = 0.1 * hybrid_score + 0.9 * normalized_score
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            print(f"\n상위 5개 최종 점수:")
            for i, cand in enumerate(candidates[:5], 1):
                meta = cand.get('metadata', {})
                if 'metadata' in meta:
                    inner_meta = meta['metadata']
                else:
                    inner_meta = meta
                course_name = inner_meta.get('강좌명', 'unknown')
                print(f"  {i}. {course_name[:20]:20s} | Hybrid: {cand.get('hybrid_score', 0):.3f}, Rerank: {cand['rerank_score']:.3f}, 최종: {cand['score']:.3f}")
            
            return candidates
            
        except Exception as e:
            print(f"Cross-Encoder 재랭킹 실패: {e}. 원본 점수 사용.")
            import traceback
            traceback.print_exc()
            return candidates

    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._rerank_with_cross_encoder(query, candidates)

    def _build_prompt(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        
        if chat_history is None:
            chat_history = []
        
        context_str_list = []
        for i, c in enumerate(contexts, start=1):
            metadata = c.get("metadata", {})
            
            if 'metadata' in metadata and isinstance(metadata['metadata'], dict):
                inner_metadata = metadata['metadata']
            else:
                inner_metadata = metadata
            
            course_name = inner_metadata.get('강좌명', 'unknown')
            print(f"  {i}. 강좌명: {course_name}")
            
            text_parts = []
            
            important_fields = [
                '강좌명', '담당교수', '교과목 개요', '교육목표', 
                '주요교재', '참고교재(대표)', '평가항목', 
                '수강대상학과', '학점', '주당시간', '강좌형식',
                '필수 선수과목', '권장 선수과목',
                '학습준비사항', '수강학생 유의 및 참고사항',
                '주차별 강의개요'
            ]
            
            for key in important_fields:
                if key in inner_metadata and inner_metadata[key]:
                    value = inner_metadata[key]
                    if isinstance(value, list):
                        if len(value) > 0:
                            text_parts.append(f"{key}:")
                            for item in value:
                                text_parts.append(f"  - {item}")
                    elif isinstance(value, dict):
                        text_parts.append(f"{key}: {value}")
                    else:
                        text_parts.append(f"{key}: {value}")
            
            for key, value in inner_metadata.items():
                if key not in important_fields and key not in ['chunk_id', 'source_pdf'] and value:
                    if isinstance(value, (str, int, float)):
                        text_parts.append(f"{key}: {value}")
            
            text_content = "\n".join(text_parts)
            
            source_pdf = metadata.get('source_pdf', c.get('source_pdf', 'unknown'))
            
            context_str_list.append(
                f"[문서 {i}] (출처: {source_pdf})\n{text_content}"
            )
        
        context_block = "\n\n---\n\n".join(context_str_list)

        system_msg = (
            "너는 숭실대학교 강의계획서 RAG 챗봇이야.\n"
            "아래 제공된 강의계획서 청크 내용만을 근거로 답변하고, "
            "모르는 내용은 아는 척 하지 말고 모른다고 말해.\n"
            "답변할 때는 강좌명, 담당교수, 교육목표, 교재 등 구체적인 정보를 포함해서 자세히 설명해줘.\n"
            "이전 대화 내용을 참고하여 맥락을 이해하고 답변해줘."
        )

        # 메시지 구성: 시스템 메시지 + 이전 대화 히스토리 + 현재 질문
        messages = [{"role": "system", "content": system_msg}]
        
        # 이전 대화 히스토리 추가 (최근 5개만)
        recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
        messages.extend(recent_history)
        
        # 현재 질문과 컨텍스트 추가
        user_msg = (
            f"다음은 강의계획서에서 추출한 관련 내용이야:\n\n"
            f"{context_block}\n\n"
            f"위 내용을 참고해서 아래 질문에 한국어로 자세히 답변해줘.\n"
            f"질문: {query}"
        )
        messages.append({"role": "user", "content": user_msg})

        return messages

    def ask(self, query: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []
            
        transformed_query = self._transform_query(query) if self.use_query_expansion else query
        filters = self._extract_metadata_filters(query)
        contexts = self._retrieve(query, transformed_query=transformed_query)
        messages = self._build_prompt(query, contexts, chat_history=chat_history)

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
        vector_results = self.vector_store.similarity_search_with_score(query, k=top_k*2)
        
        for doc, score in vector_results:
            metadata = doc.metadata
            content = doc.page_content.lower()
            
            if any(keyword in content for keyword in ["교재", "참고도서", "참고문헌", "textbook"]):
                score *= 1.2

def get_relevant_contexts(
    query: str,
    vectorstore,
    bm25_index,
    reranker,
    k: int = 20,
    final_k: int = 5
) -> List[Dict]:
    query_expansion = {
        "주요교재": ["교재", "주교재", "참고서적", "교과서", "textbook", "참고도서"],
        "참고교재": ["부교재", "참고서적", "추천도서", "reference book"],
        "교수": ["담당교수", "교수님", "강의자", "instructor"],
        "학점": ["성적", "평가", "점수", "grade"]
    }
    
    expanded_query = query
    for key, synonyms in query_expansion.items():
        if key in query:
            expanded_query = query + " " + " ".join(synonyms)
            break
    
    vector_results = vectorstore.similarity_search_with_score(expanded_query, k=k)
    
    bm25_scores = bm25_index["index"].get_scores(
        bm25_index["tokenizer"](expanded_query)
    )