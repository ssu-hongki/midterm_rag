# -*- coding: utf-8 -*-

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio
from typing import Optional
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# 작업 디렉토리를 프로젝트 루트로 변경
os.chdir(PROJECT_ROOT)

from rag_chain import RAGChain
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="강의계획서 RAG API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG Chain 전역 인스턴스
rag_chain: Optional[RAGChain] = None


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class QueryRequest(BaseModel):
    query: str
    use_reranking: bool = True
    use_query_expansion: bool = True
    chat_history: list[ChatMessage] = []  # 이전 대화 히스토리


class ConfigRequest(BaseModel):
    use_reranking: bool = True
    use_query_expansion: bool = True


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 RAG Chain 초기화"""
    global rag_chain
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
        return
    
    try:
        rag_chain = RAGChain(k=5, use_reranking=True, use_query_expansion=True)
        print("✅ RAG Chain 초기화 완료")
    except Exception as e:
        print(f"❌ RAG Chain 초기화 실패: {e}")


@app.get("/")
async def root():
    return {
        "message": "강의계획서 RAG API",
        "status": "running",
        "rag_chain_initialized": rag_chain is not None
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "rag_chain_ready": rag_chain is not None
    }


@app.post("/api/config")
async def update_config(config: ConfigRequest):
    """RAG 설정 업데이트"""
    global rag_chain
    
    try:
        rag_chain = RAGChain(
            k=5,
            use_reranking=config.use_reranking,
            use_query_expansion=config.use_query_expansion
        )
        return {
            "status": "success",
            "config": {
                "use_reranking": config.use_reranking,
                "use_query_expansion": config.use_query_expansion
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 업데이트 실패: {str(e)}")


async def generate_rag_response(query: str, chat_history: list = None):
    """
    RAG 응답을 스트리밍으로 생성
    Server-Sent Events (SSE) 형식으로 진행 상황 전달
    """
    global rag_chain
    
    if chat_history is None:
        chat_history = []
    
    if rag_chain is None:
        yield f"data: {json.dumps({'type': 'error', 'message': 'RAG Chain이 초기화되지 않았습니다.'}, ensure_ascii=False)}\n\n"
        return
    
    try:
        # 즉각적인 배경 지식 제공 (미니 답변)
        yield f"data: {json.dumps({'type': 'preview_start'}, ensure_ascii=False)}\n\n"
        
        # 질문에서 주요 키워드 추출하여 간단한 배경 지식 생성
        preview_prompt = f"""사용자가 '{query}'에 대해 질문했어.
이 질문의 주제에 대한 간단하고 흥미로운 배경 지식을 2-3문장으로 알려줘.
구체적인 강의 내용은 언급하지 말고, 일반적인 개념이나 흥미로운 사실만 제공해줘.
친근한 반말로 말해줘."""

        preview_stream = rag_chain.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 친근하고 재밌는 친구같은 AI야. 반말로 편하게 대화하고, 사용자의 질문 주제에 대한 흥미로운 배경 지식을 알려줘. 이모티콘은 사용하지 마."},
                {"role": "user", "content": preview_prompt}
            ],
            max_tokens=150,
            temperature=0.8,
            stream=True
        )
        
        for chunk in preview_stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'type': 'preview_chunk', 'content': content}, ensure_ascii=False)}\n\n"
        
        yield f"data: {json.dumps({'type': 'preview_complete'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.5)
        
        # 이제 실제 RAG 검색 시작
        yield f"data: {json.dumps({'type': 'rag_start'}, ensure_ascii=False)}\n\n"
        
        # 1단계: 질문 분석 시작
        yield f"data: {json.dumps({'type': 'status', 'step': 'analyzing', 'message': '질문 분석 중...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)
        
        # 질문 변환
        transformed_query = query
        if rag_chain.use_query_expansion:
            transformed_query = rag_chain._transform_query(query)
            if transformed_query != query:
                yield f"data: {json.dumps({'type': 'transformed_query', 'original': query, 'transformed': transformed_query}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.2)
        
        # 2단계: 메타데이터 필터 추출
        yield f"data: {json.dumps({'type': 'status', 'step': 'filtering', 'message': '관련 강의 필터링 중...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)
        
        filters = rag_chain._extract_metadata_filters(query)
        if filters:
            yield f"data: {json.dumps({'type': 'filters', 'filters': filters}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.2)
        
        # 3단계: 문서 검색
        yield f"data: {json.dumps({'type': 'status', 'step': 'searching', 'message': '관련 문서 검색 중...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)
        
        # Query expansion
        if rag_chain.use_query_expansion:
            expanded_queries = rag_chain._expand_query(transformed_query or query)
            yield f"data: {json.dumps({'type': 'expanded_queries', 'queries': expanded_queries}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.2)
        
        # 실제 검색 수행
        contexts = rag_chain._retrieve(query, transformed_query=transformed_query)
        yield f"data: {json.dumps({'type': 'contexts_found', 'count': len(contexts)}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.3)
        
        # 4단계: 재랭킹
        if rag_chain.use_reranking:
            yield f"data: {json.dumps({'type': 'status', 'step': 'reranking', 'message': '결과 재정렬 중...'}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.3)
        
        # 5단계: 답변 생성
        yield f"data: {json.dumps({'type': 'status', 'step': 'generating', 'message': '답변 생성 중...'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.2)
        
        # 프롬프트 생성 (대화 히스토리 포함)
        messages = rag_chain._build_prompt(query, contexts, chat_history=chat_history)
        
        # OpenAI API 호출 (스트리밍)
        stream = rag_chain.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )
        
        # 답변 스트리밍 시작
        yield f"data: {json.dumps({'type': 'answer_start'}, ensure_ascii=False)}\n\n"
        
        full_answer = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_answer += content
                yield f"data: {json.dumps({'type': 'answer_chunk', 'content': content}, ensure_ascii=False)}\n\n"
        
        # 답변 완료
        yield f"data: {json.dumps({'type': 'answer_complete', 'full_answer': full_answer}, ensure_ascii=False)}\n\n"
        
        # 컨텍스트 정보 전달
        context_data = []
        for ctx in contexts[:5]:
            meta = ctx.get("metadata", {})
            if 'metadata' in meta:
                inner_meta = meta['metadata']
            else:
                inner_meta = meta
            
            context_data.append({
                "강좌명": inner_meta.get("강좌명", ""),
                "과목코드": inner_meta.get("과목코드", ""),
                "담당교수": inner_meta.get("담당교수", ""),
                "source_pdf": meta.get("source_pdf", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "score": ctx.get("score", 0),
                "text_preview": str(inner_meta)[:200] + "..."
            })
        
        yield f"data: {json.dumps({'type': 'contexts', 'contexts': context_data}, ensure_ascii=False)}\n\n"
        
        # 완료
        yield f"data: {json.dumps({'type': 'complete'}, ensure_ascii=False)}\n\n"
        
    except Exception as e:
        error_message = f"오류 발생: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'message': error_message}, ensure_ascii=False)}\n\n"


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """
    스트리밍 방식으로 질문에 답변
    Server-Sent Events (SSE) 사용
    """
    global rag_chain
    
    # 설정이 변경되었으면 RAG Chain 재초기화
    if rag_chain is None or \
       rag_chain.use_reranking != request.use_reranking or \
       rag_chain.use_query_expansion != request.use_query_expansion:
        try:
            rag_chain = RAGChain(
                k=5,
                use_reranking=request.use_reranking,
                use_query_expansion=request.use_query_expansion
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG Chain 초기화 실패: {str(e)}")
    
    # 대화 히스토리를 dict 리스트로 변환
    history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
    
    return StreamingResponse(
        generate_rag_response(request.query, chat_history=history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # nginx 버퍼링 비활성화
        }
    )


@app.post("/api/query")
async def query_simple(request: QueryRequest):
    """
    일반 방식으로 질문에 답변 (스트리밍 없음)
    """
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG Chain이 초기화되지 않았습니다.")
    
    # 설정이 변경되었으면 RAG Chain 재초기화
    if rag_chain.use_reranking != request.use_reranking or \
       rag_chain.use_query_expansion != request.use_query_expansion:
        try:
            rag_chain = RAGChain(
                k=5,
                use_reranking=request.use_reranking,
                use_query_expansion=request.use_query_expansion
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG Chain 초기화 실패: {str(e)}")
    
    try:
        # 대화 히스토리를 dict 리스트로 변환
        history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]
        result = rag_chain.ask(request.query, chat_history=history)
        
        # 컨텍스트 단순화
        simplified_contexts = []
        for ctx in result.get("contexts", [])[:5]:
            meta = ctx.get("metadata", {})
            if 'metadata' in meta:
                inner_meta = meta['metadata']
            else:
                inner_meta = meta
            
            simplified_contexts.append({
                "강좌명": inner_meta.get("강좌명", ""),
                "과목코드": inner_meta.get("과목코드", ""),
                "담당교수": inner_meta.get("담당교수", ""),
                "source_pdf": meta.get("source_pdf", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "score": ctx.get("score", 0)
            })
        
        return {
            "answer": result["answer"],
            "contexts": simplified_contexts,
            "original_query": result.get("original_query"),
            "transformed_query": result.get("transformed_query"),
            "metadata_filters": result.get("metadata_filters")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질문 처리 실패: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

