import json
import os
from langchain_core.documents import Document
from typing import List, Dict, Any

JSON_DATA_PATH = "data/all_syllabi_rag_data.json"

def load_json_docs() -> List[Document]:
    
    if not os.path.exists(JSON_DATA_PATH):
        print(f"❌ 오류: '{JSON_DATA_PATH}' 파일을 찾을 수 없습니다.")
        print("  [해결책] 먼저 'preprocess_pdfs.py' 스크립트를 실행하여 JSON 파일을 생성해야 합니다.")
        print("  ( 'data/pdfs/' 폴더에 PDF 파일들을 넣은 후, 'python preprocess_pdfs.py' 실행 )")
        return []

    print(f"➡️ '{JSON_DATA_PATH}' 파일에서 전처리된 데이터를 로드합니다...")

    try:
        with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
            all_rag_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 오류: '{JSON_DATA_PATH}' 파일이 올바른 JSON 형식이 아닙니다. {e}")
        return []
    except Exception as e:
        print(f"❌ 오류: '{JSON_DATA_PATH}' 파일 읽기 중 오류 발생: {e}")
        return []

    all_documents = []
    
    for chunk in all_rag_data:
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        
        page_content = f"## {title} ##\n{content}"
        
        doc_metadata = metadata.copy()
        
        document = Document(page_content=page_content, metadata=doc_metadata)
        all_documents.append(document)

    print(f"  [성공] 총 {len(all_documents)}개의 데이터 청크(Document)를 로드했습니다.")
    return all_documents