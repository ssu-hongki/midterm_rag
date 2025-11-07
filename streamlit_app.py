import os

import streamlit as st
from dotenv import load_dotenv

import data_loader
import rag_chain
import vector_store


CHROMA_PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "syllabus_rag_json"


def _initialize_resources(rebuild: bool = False) -> None:
    """세션 상태에 벡터스토어 및 RAG 체인을 준비합니다."""

    documents = None
    if rebuild:
        with st.spinner("JSON 문서 로드 중..."):
            documents = data_loader.load_json_docs()
        if not documents:
            st.error("JSON 문서를 불러오지 못했습니다. 'preprocess_pdfs.py' 실행 여부를 확인하세요.")
            return

    with st.spinner("벡터스토어 초기화 중..."):
        vectordb = vector_store.get_vector_store(
            documents=documents,
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            rebuild=rebuild,
        )

    if vectordb is None:
        if rebuild:
            st.error("벡터스토어 재구축에 실패했습니다. 터미널 출력을 확인하세요.")
        else:
            st.error("기존 벡터스토어를 로드하지 못했습니다. 먼저 재구축하세요.")
        return

    try:
        chain = rag_chain.build_rag_chain(vectordb)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"RAG 체인 구성 실패: {exc}")
        return

    st.session_state["vectordb"] = vectordb
    st.session_state["rag_chain"] = chain


def _ensure_resources_initialized() -> None:
    if "rag_chain" not in st.session_state or st.session_state.get("rag_chain") is None:
        _initialize_resources(rebuild=False)


def _handle_query(query: str) -> None:
    chain = st.session_state.get("rag_chain")
    if chain is None:
        st.warning("RAG 체인이 초기화되지 않았습니다. 벡터스토어를 먼저 준비하세요.")
        return

    with st.spinner("답변 생성 중..."):
        result = chain.invoke(query)

    answer = result.get("answer", "답변을 생성하지 못했습니다.")
    source_docs = result.get("source_docs", [])

    st.markdown("### 💬 답변")
    st.write(answer)

    if source_docs:
        st.markdown("### 📚 출처")
        seen_sources = set()
        for doc in source_docs:
            metadata = doc.metadata
            source_file = metadata.get("source_file") or metadata.get("source") or "N/A"
            title = doc.page_content.split("\n", maxsplit=1)[0].replace("##", "").strip()
            key = (source_file, title)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            st.write(f"- **{source_file}** · {title}")


def main() -> None:
    st.set_page_config(
        page_title="강의계획서 RAG 챗봇",
        page_icon="📘",
    )

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. 프로젝트 루트의 .env 파일을 확인하세요.")
        st.stop()

    st.sidebar.title("설정")
    st.sidebar.info("필요 시 벡터스토어를 재구축할 수 있습니다.")

    st.sidebar.button(
        "벡터스토어 재구축",
        on_click=_initialize_resources,
        kwargs={"rebuild": True},
        type="primary",
    )

    _ensure_resources_initialized()

    st.title("📘 강의계획서 RAG 챗봇")
    st.write("강의계획서 JSON 데이터에 기반하여 질문에 답변합니다.")

    with st.form(key="query_form"):
        query = st.text_area("질문을 입력하세요", height=120, placeholder="예: 3주차 실습 내용이 뭐야?")
        submitted = st.form_submit_button("질문하기")

    if submitted:
        if query.strip():
            _handle_query(query.strip())
        else:
            st.warning("질문을 입력해 주세요.")


if __name__ == "__main__":
    main()


