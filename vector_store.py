import os
import shutil
from langchain_openai import OpenAIEmbeddings
# [수정됨] langchain_community.vectorstores 대신 최신 langchain_chroma를 임포트
from langchain_chroma import Chroma 
from langchain_core.documents import Document
from typing import List, Optional

def get_vector_store(
    documents: Optional[List[Document]] = None, 
    persist_dir: str = "data/chroma",
    collection_name: str = "syllabus_rag_json",
    rebuild: bool = False
) -> Optional[Chroma]:
    """
    Chroma 벡터스토어를 생성하거나 로드합니다.
    - rebuild=True: documents를 기반으로 DB를 새로 생성합니다.
    - rebuild=False: persist_dir에서 기존 DB를 로드합니다.
    """
    
    # 1. 임베딩 모델 정의 (한 번만 로드)
    try:
        # text-embedding-3-small이 비용 효율적이고 성능이 좋음
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    except Exception as e:
        print(f"❌ 오류: OpenAI 임베딩 모델 로드 실패. API 키를 확인하세요. {e}")
        return None

    # 2. DB 새로 생성 (rebuild=True)
    if rebuild:
        if not documents:
            print("❌ 오류: DB를 새로 생성하려면 'documents' 리스트가 필요합니다.")
            return None
            
        print("기존 벡터스토어를 삭제하고 새로 생성합니다...")
        
        # 기존 폴더 삭제
        if os.path.exists(persist_dir):
            try:
                shutil.rmtree(persist_dir)
                print(f"  - 기존 '{persist_dir}' 폴더 삭제 완료")
            except OSError as e:
                print(f"❌ 오류: 기존 '{persist_dir}' 폴더 삭제 실패: {e}")
                return None
        
        # 벡터스토어 생성 및 저장
        try:
            # [수정됨] 임포트된 최신 Chroma 클래스 사용
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=embedding,
                collection_name=collection_name,
                persist_directory=persist_dir
            )
            print("벡터스토어 생성 완료.")
            return vectordb
        except Exception as e:
            print(f"❌ 오류: Chroma DB 생성 실패: {e}")
            return None

    # 3. 기존 DB 로드 (rebuild=False)
    else:
        if not os.path.exists(persist_dir):
            print(f"❌ 오류: '{persist_dir}' 폴더를 찾을 수 없습니다. (rebuild=False)")
            return None
            
        try:
            # [수정됨] 임포트된 최신 Chroma 클래스 사용
            vectordb = Chroma(
                persist_directory=persist_dir,
                collection_name=collection_name,
                embedding_function=embedding
            )
            return vectordb
        except Exception as e:
            print(f"❌ 오류: Chroma DB 로드 실패: {e}")
            return None