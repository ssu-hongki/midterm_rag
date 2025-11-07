import os
import argparse
import shutil
from dotenv import load_dotenv

# [JSON 파이프라인으로 변경]
# 1. data_loader는 이제 load_json_docs()를 호출합니다.
# 2. chunking.py는 더 이상 필요 없으므로 import 및 호출 코드를 모두 삭제합니다.
import data_loader 
import vector_store
import rag_chain

# --- 상수 정의 ---
# (DATA_DIR은 data_loader.py에 하드코딩되었으므로 삭제)
CHROMA_PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "syllabus_rag_json" # (이전 컬렉션과 분리)

def main():
    """
    메인 RAG 파이프라인 실행 함수
    1. (JSON 로드) -> 2. (임베딩/저장) -> 3. (RAG 체인) -> 4. (질의응답)
    """
    
    # 0. .env 파일 로드 (OPENAI_API_KEY)
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
        return

    # 1. 인자 파서 설정
    parser = argparse.ArgumentParser(description="강의계획서 JSON RAG 시스템")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="기존 ChromaDB를 삭제하고 벡터스토어를 새로 생성합니다."
    )
    args = parser.parse_args()

    vectordb = None

    if args.rebuild:
        print("--- [RAG 시스템 빌드 시작 (JSON)] ---")
        
        # --- 1. JSON 파일 로드 ---
        print("\n1. 전처리된 JSON 파일 로드 중...")
        # data_loader.py의 새 함수 호출 (인자 필요 없음)
        documents = data_loader.load_json_docs() 
        
        if not documents:
            print("❌ 오류: JSON에서 문서를 로드하지 못했습니다.")
            print("  [해결책] 'python preprocess_pdfs.py'를 먼저 실행했는지 확인하세요.")
            return

        # --- [변경] 2. 청크화 단계 (삭제) ---
        # JSON 파일 자체가 이미 완벽하게 청크화되어 있으므로,
        # chunking.py를 호출할 필요가 없습니다.
        # 'documents' 변수가 'chunks' 변수와 동일한 역할을 합니다.
        print(f"\n2. (청크화 단계 생략 - {len(documents)}개의 청크 로드 완료)")

        # --- 3. 벡터 임베딩 및 ChromaDB 저장 ---
        print("\n3. 벡터 임베딩 및 ChromaDB 저장 중...")
        vectordb = vector_store.get_vector_store(
            documents, # 'chunks' 대신 'documents' (이미 청크임)
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            rebuild=True
        )
        print("\n--- [RAG 시스템 빌드 완료] ---")
        
    else:
        # --- 기존 벡터스토어 로드 ---
        print("--- [RAG 시스템 로드 중 (JSON)] ---")
        if not os.path.exists(CHROMA_PERSIST_DIR):
            print(f"❌ 오류: '{CHROMA_PERSIST_DIR}' 폴더를 찾을 수 없습니다.")
            print("  [해결책] 먼저 '--rebuild' 옵션으로 시스템을 빌드하세요.")
            print("  (예: python main_xls_rag.py --rebuild)")
            return
            
        vectordb = vector_store.get_vector_store(
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            rebuild=False
        )
        if vectordb is None:
             print("❌ 오류: 벡터스토어 로드에 실패했습니다.")
             return
        print("  [성공] 기존 벡터스토어 로드 완료.")


    # --- 4. RAG 체인 구성 ---
    try:
        rag_chain_instance = rag_chain.build_rag_chain(vectordb)
    except Exception as e:
        print(f"❌ RAG 체인 구성 중 오류 발생: {e}")
        return

    # --- 5. 콘솔 질의응답 루프 ---
    print("\n" + "-"*50)
    print("강의계획서 RAG 시스템이 준비되었습니다. (JSON 기반)")
    print("질문을 입력하세요. (종료: 'exit' 또는 'q')")
    print("-"*50)

    while True:
        try:
            query = input("질문 입력 > ")
            if query.lower() in ['exit', 'q', 'quit']:
                print("시스템을 종료합니다.")
                break
            if not query.strip():
                continue

            # RAG 체인 실행
            result = rag_chain_instance.invoke(query)
            
            answer = result.get("answer", "답변을 생성하지 못했습니다.")
            source_docs = result.get("source_docs", [])

            print("\n💬 답변:")
            print(answer)

            # (선택적) 출처 표시
            if source_docs:
                print("\n📚 출처 (메타데이터):")
                # (중복 출처 제거)
                seen_sources = set()
                for doc in source_docs:
                    source = doc.metadata.get('source_file', 'N/A')
                    title = doc.page_content.split('\n')[0].replace("##", "").strip()
                    # (예: [DB 강의계획서 2025.pdf] - 주차별 강의개요: 3주차)
                    source_str = f"  - [{source}] - {title}"
                    if source_str not in seen_sources:
                        print(source_str)
                        seen_sources.add(source_str)

            print("\n" + "-"*50)

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n강제 종료...")
            break
        except Exception as e:
            print(f"\n[!] 답변 생성 중 오류가 발생했습니다: {e}")
            # (오류 발생 시 다음 질문을 받을 수 있도록 루프 계속)


if __name__ == "__main__":
    main()