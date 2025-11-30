# -*- coding: utf-8 -*-

from pathlib import Path
from preprocess_pdfs import process_all_pdfs
from data_loader import load_documents
from vector_store import build_vector_store
from rag_chain import RAGChain

def ensure_preprocessed():
    json_path = Path("data/processed/course_chunks.json")
    if not json_path.exists():
        process_all_pdfs()

def ensure_vector_store():
    store_dir = Path("data/vector_store")
    emb_path = store_dir / "embeddings.npy"
    meta_path = store_dir / "metadatas.json"

    if not emb_path.exists() or not meta_path.exists():
        docs = load_documents()
        build_vector_store(docs, store_dir=store_dir)

def interactive_chat():
    rag = RAGChain(k=5)
    print("\n==============================")
    print("강의계획서 RAG 챗봇 (종료: quit / exit)")
    print("==============================\n")

    while True:
        q = input("질문 > ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue

        result = rag.ask(q)
        print("\n[답변]")
        print(result["answer"])

        print("\n" + "=" * 40 + "\n")

if __name__ == "__main__":
    ensure_preprocessed()
    ensure_vector_store()
    interactive_chat()