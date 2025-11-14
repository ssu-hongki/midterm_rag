# data_loader.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Dict, Any

@dataclass
class Document:
    id: str        # "파일명_chunkid" 형태
    text: str
    metadata: Dict[str, Any]

def load_documents(
    json_path: Path = Path("data/processed/course_chunks.json")
) -> List[Document]:
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} 가 존재하지 않습니다. 먼저 preprocess_pdfs.py를 실행하세요.")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs: List[Document] = []
    for idx, item in enumerate(raw):
        doc_id = f"{item.get('source_pdf','unknown')}_chunk{item.get('chunk_id', idx+1)}"
        text = item["text"]
        meta = dict(item.get("metadata", {}))
        meta["source_pdf"] = item.get("source_pdf")
        meta["chunk_id"] = item.get("chunk_id")
        docs.append(Document(id=doc_id, text=text, metadata=meta))

    print(f"📚 Loaded {len(docs)} documents from {json_path}")
    return docs