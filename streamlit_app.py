# streamlit_app.py
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from data_loader import load_documents
from vector_store import build_vector_store, load_vector_store
from rag_chain import RAGChain

VECTOR_STORE_DIR = Path("data/vector_store")
PROCESSED_JSON_PATH = Path("data/processed/course_chunks.json")


def ensure_vector_store_exists():
    """벡터 스토어가 존재하는지 확인하고, 없으면 생성합니다."""
    emb_path = VECTOR_STORE_DIR / "embeddings.npy"
    meta_path = VECTOR_STORE_DIR / "metadatas.json"
    
    if not emb_path.exists() or not meta_path.exists():
        return False
    return True


def initialize_rag_chain(use_reranking: bool = True, use_query_expansion: bool = True):
    """RAG 체인을 초기화하고 세션 상태에 저장합니다."""
    # 설정이 변경되었거나 체인이 없으면 재초기화
    current_reranking = st.session_state.get("use_reranking", True)
    current_expansion = st.session_state.get("use_query_expansion", True)
    if ("rag_chain" in st.session_state and 
        st.session_state["rag_chain"] is not None and 
        current_reranking == use_reranking and
        current_expansion == use_query_expansion):
        return
    
    with st.spinner("벡터 스토어 로드 중..."):
        try:
            rag_chain = RAGChain(k=5, use_reranking=use_reranking, use_query_expansion=use_query_expansion)
            st.session_state["rag_chain"] = rag_chain
            st.session_state["use_reranking"] = use_reranking
            st.session_state["use_query_expansion"] = use_query_expansion
        except FileNotFoundError as e:
            st.error(f"벡터 스토어를 찾을 수 없습니다: {e}")
            st.info("먼저 벡터 스토어를 구축해주세요.")
            st.stop()
        except Exception as e:
            st.error(f"RAG 체인 초기화 실패: {e}")
            st.stop()


def build_vector_store_from_documents():
    """문서를 로드하고 벡터 스토어를 구축합니다."""
    if not PROCESSED_JSON_PATH.exists():
        st.error(f"전처리된 JSON 파일이 없습니다: {PROCESSED_JSON_PATH}")
        st.info("먼저 'preprocess_pdfs.py'를 실행하여 PDF를 전처리해주세요.")
        return False
    
    with st.spinner("문서 로드 중..."):
        try:
            documents = load_documents(PROCESSED_JSON_PATH)
        except Exception as e:
            st.error(f"문서 로드 실패: {e}")
            return False
    
    with st.spinner("벡터 스토어 구축 중... (시간이 걸릴 수 있습니다)"):
        try:
            build_vector_store(documents, store_dir=VECTOR_STORE_DIR)
            st.success("벡터 스토어 구축 완료!")
            # 세션 상태 초기화하여 다시 로드하도록 함
            if "rag_chain" in st.session_state:
                del st.session_state["rag_chain"]
            return True
        except Exception as e:
            st.error(f"벡터 스토어 구축 실패: {e}")
            return False


def handle_query(query: str):
    """질문을 처리하고 답변을 표시합니다."""
    rag_chain = st.session_state.get("rag_chain")
    if rag_chain is None:
        st.warning("RAG 체인이 초기화되지 않았습니다.")
        return
    
    with st.spinner("답변 생성 중..."):
        try:
            result = rag_chain.ask(query)
            
            # 변환된 질문 및 필터 정보 표시
            if result.get("transformed_query") or result.get("metadata_filters"):
                with st.expander("🔍 검색 정보", expanded=False):
                    if result.get("transformed_query"):
                        st.markdown(f"**원본 질문:** {result.get('original_query')}")
                        st.markdown(f"**변환된 질문:** {result.get('transformed_query')}")
                        st.caption("질문이 더 명확하고 검색하기 좋은 형태로 자동 변환되었습니다.")
                    
                    if result.get("metadata_filters"):
                        st.markdown("**적용된 필터:**")
                        filters = result.get("metadata_filters")
                        filter_items = []
                        if filters.get("수강대상학과"):
                            filter_items.append(f"수강대상학과: {filters['수강대상학과']}")
                        if filters.get("학년"):
                            filter_items.append(f"학년: {filters['학년']}")
                        if filters.get("강좌명") or filters.get("강좌명_키워드"):
                            강좌명 = filters.get("강좌명") or filters.get("강좌명_키워드")
                            filter_items.append(f"강좌명: {강좌명}")
                        if filters.get("담당교수"):
                            filter_items.append(f"담당교수: {filters['담당교수']}")
                        if filters.get("과목코드"):
                            filter_items.append(f"과목코드: {filters['과목코드']}")
                        
                        if filter_items:
                            for item in filter_items:
                                st.markdown(f"- {item}")
                        st.caption("metadata 필터링이 적용되어 관련 강의만 검색되었습니다.")
            
            # 답변 표시
            st.markdown("### 💬 답변")
            st.write(result["answer"])
            
            # 출처 표시
            contexts = result.get("contexts", [])
            if contexts:
                st.markdown("### 📚 참고한 컨텍스트")
                seen_sources = set()
                for i, ctx in enumerate(contexts[:5], start=1):
                    meta = ctx.get("metadata", {}) or {}
                    source_pdf = meta.get("source_pdf", "unknown")
                    chunk_id = meta.get("chunk_id", "unknown")
                    강좌명 = meta.get("강좌명", "")
                    과목코드 = meta.get("과목코드", "")
                    
                    # 중복 제거
                    key = (source_pdf, chunk_id)
                    if key in seen_sources:
                        continue
                    seen_sources.add(key)
                    
                    score = ctx.get("score", None)
                    source_info = f"**{source_pdf}**"
                    if 강좌명:
                        source_info += f" · {강좌명}"
                    if 과목코드:
                        source_info += f" ({과목코드})"
                    source_info += f" · 청크 {chunk_id}"
                    if score is not None:
                        source_info += f" · 유사도: {score:.3f}"
                    
                    # 청크 내용도 함께 표시
                    with st.expander(f"{i}. {source_info}", expanded=False):
                        chunk_text = ctx.get("text", "내용 없음")
                        st.markdown("**청크 내용:**")
                        st.text_area(
                            "",
                            value=chunk_text,
                            height=150,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"chunk_{i}_{chunk_id}_{hash(chunk_text)}"
                        )
                        
                        # 매칭된 질문들 표시 (확장된 경우)
                        if ctx.get("matched_queries") and len(ctx.get("matched_queries", [])) > 1:
                            st.caption(f"매칭된 질문: {', '.join(ctx['matched_queries'][:2])}")
        except Exception as e:
            st.error(f"질문 처리 중 오류 발생: {e}")


def main():
    st.set_page_config(
        page_title="강의계획서 RAG 챗봇",
        page_icon="📘",
        layout="wide"
    )
    
    # 환경 변수 로드
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY가 설정되어 있지 않습니다.")
        st.info("프로젝트 루트에 .env 파일을 생성하고 OPENAI_API_KEY를 설정해주세요.")
        st.stop()
    
    # 사이드바
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # 벡터 스토어 상태 확인
        if ensure_vector_store_exists():
            st.success("✅ 벡터 스토어가 준비되어 있습니다.")
        else:
            st.warning("⚠️ 벡터 스토어가 없습니다.")
            if st.button("🔨 벡터 스토어 구축", type="primary"):
                if build_vector_store_from_documents():
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔍 검색 설정")
        use_reranking = st.checkbox(
            "재랭킹 사용 (Reranking)",
            value=True,
            help="Cross-encoder를 사용하여 검색 결과를 더 정확하게 재정렬합니다. 정확도는 높아지지만 속도가 약간 느려집니다."
        )
        use_query_expansion = st.checkbox(
            "질문 변환/확장 사용 (Query Transformation/Expansion)",
            value=True,
            help="질문을 자동으로 명확하게 변환하고 여러 관점에서 확장하여 검색합니다. 오타나 모호한 표현도 잘 처리합니다."
        )
        
        st.markdown("---")
        st.markdown("### 📋 사용 방법")
        st.markdown("""
        1. 벡터 스토어가 준비되어 있는지 확인하세요
        2. 질문을 입력하고 '질문하기' 버튼을 클릭하세요
        3. 답변과 참고한 컨텍스트를 확인하세요
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ 정보")
        st.markdown("""
        - 이 챗봇은 강의계획서 PDF를 전처리한 데이터를 기반으로 답변합니다
        - 벡터 스토어가 없으면 먼저 구축해야 합니다
        - 재랭킹 기능은 검색 정확도를 향상시킵니다
        """)
    
    # 메인 영역
    st.title("📘 강의계획서 RAG 챗봇")
    st.markdown("강의계획서 데이터를 기반으로 질문에 답변하는 챗봇입니다.")
    
    # RAG 체인 초기화 (설정 적용)
    initialize_rag_chain(use_reranking=use_reranking, use_query_expansion=use_query_expansion)
    
    # 질문 입력 폼
    with st.form(key="query_form", clear_on_submit=False):
        query = st.text_area(
            "질문을 입력하세요",
            height=120,
            placeholder="예: 자연언어처리 3주차 실습 내용이 뭐야?\n예: 자연언어처리 강의의 평가 방법은?\n예: 자연언어처리 강의의 교재는?\n\n💡 팁: 오타나 모호한 표현도 자동으로 처리됩니다!\n예: 'nlp 3주차 뭐함?' → 자동 변환됨"
        )
        submitted = st.form_submit_button("질문하기", type="primary", use_container_width=True)
    
    # 질문 처리
    if submitted:
        if query.strip():
            handle_query(query.strip())
        else:
            st.warning("⚠️ 질문을 입력해주세요.")


if __name__ == "__main__":
    main()