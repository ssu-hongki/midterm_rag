# main_rag.py
# -*- coding: utf-8 -*-

from pathlib import Path

from preprocess_pdfs import process_all_pdfs
from data_loader import load_documents
from vector_store import build_vector_store
from rag_chain import RAGChain

def ensure_preprocessed():
    json_path = Path("data/processed/course_chunks.json")
    if not json_path.exists():
        print("📄 전처리된 JSON이 없습니다. PDF를 먼저 처리합니다.")
        process_all_pdfs()
    else:
        print("📄 전처리된 JSON이 이미 존재합니다. (건너뜀)")

def ensure_vector_store():
    store_dir = Path("data/vector_store")
    emb_path = store_dir / "embeddings.npy"
    meta_path = store_dir / "metadatas.json"

    if not emb_path.exists() or not meta_path.exists():
        print("🧠 벡터 스토어가 없습니다. 새로 생성합니다.")
        docs = load_documents()
        build_vector_store(docs, store_dir=store_dir)
    else:
        print("🧠 벡터 스토어가 이미 존재합니다. (건너뜀)")

def interactive_chat():
    rag = RAGChain(k=5)
    print("\n==============================")
    print("강의계획서 RAG 챗봇 (종료: quit / exit)")
    print("==============================\n")

    while True:
        q = input("질문 > ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            print("👋 종료합니다.")
            break
        if not q:
            continue

        result = rag.ask(q)
        print("\n[답변]")
        print(result["answer"])

        # 🔽🔽 여기부터 출력 형식만 수정: 텍스트는 안 보여주고,
        # 어떤 파일의 몇 번 청크를 썼는지만 표시
        print("\n[참고한 컨텍스트 정보]")
        for i, ctx in enumerate(result["contexts"][:5], start=1):
            meta = ctx.get("metadata", {}) or {}
            source_pdf = meta.get("source_pdf", "unknown")
            chunk_id = meta.get("chunk_id", "unknown")
            score = ctx.get("score", None)

            line = f"- Context {i}: 파일={source_pdf}, 청크={chunk_id}"
            if score is not None:
                line += f", 유사도={score:.3f}"
            print(line)

        print("\n" + "=" * 40 + "\n")

if __name__ == "__main__":
    ensure_preprocessed()
    ensure_vector_store()
    interactive_chat()