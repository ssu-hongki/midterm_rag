# rebuild_vector_store.py
# -*- coding: utf-8 -*-
"""
벡터스토어를 강제로 재구축하는 스크립트
course_chunks.json이 업데이트된 후 실행하세요.
"""

from pathlib import Path
from data_loader import load_documents
from vector_store import build_vector_store

def rebuild_vector_store():
    """벡터스토어를 강제로 재구축합니다."""
    print("🔄 벡터스토어 재구축을 시작합니다...")
    
    # 문서 로드
    print("📚 문서 로드 중...")
    documents = load_documents()
    print(f"✅ {len(documents)}개의 문서를 로드했습니다.")
    
    # 벡터스토어 구축
    store_dir = Path("data/vector_store")
    build_vector_store(documents, store_dir=store_dir)
    
    print("\n✅ 벡터스토어 재구축이 완료되었습니다!")

if __name__ == "__main__":
    rebuild_vector_store()

