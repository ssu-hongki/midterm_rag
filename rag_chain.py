import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from utils import WEEK_NUM_REGEX, TEXTBOOK_KEYWORDS

# --- 시스템 프롬프트 정의 ---
SYSTEM_PROMPT = """너는 숭실대학교 강의계획서 전문 분석가(RAG 봇)다.
너의 임무는 오직 제공된 [강의계획서 컨텍스트]만을 근거로 사용자의 질문에 답하는 것이다.

[규칙]
1.  답변은 반드시 [강의계획서 컨텍스트]에서 찾은 내용으로만 구성한다.
2.  컨텍스트에 질문과 관련된 정보가 전혀 없으면, "강의계획서에 해당 정보가 명시되어 있지 않습니다."라고만 답변한다.
3.  절대로 외부 지식, 개인적인 의견, 추측, 또는 컨텍스트에 없는 정보를 사용해 답변을 꾸며내지 마라 (환각 금지).
4.  답변은 명확하고 간결하게 핵심 정보만 전달한다.
5.  답변 시, 가능하다면 어떤 과목의 정보인지 [과목명/담당교수/년도/학기]와 같은 메타데이터를 함께 인용하라.
6.  질문이 한국어라도 컨텍스트 내용이 영어(예: 교재명)라면, 원문 그대로 영어로 답변한다.

[강의계획서 컨텍스트]
{context}
"""

def _format_docs_with_metadata(docs: List[Document]) -> str:
    """
    검색된 Document 리스트를 메타데이터와 함께 LLM에 전달할 텍스트로 변환합니다.
    """
    if not docs:
        return "관련된 강의계획서 정보를 찾지 못했습니다."
    
    formatted_strings = []
    for doc in docs:
        # 메타데이터 포매팅
        meta = doc.metadata
        header = f"[{meta.get('course_title', 'N/A')} / {meta.get('course_code', 'N/A')} / {meta.get('semester', 'N/A')} / {meta.get('year', 'N/A')} / {meta.get('instructor', 'N/A')} / src:{meta.get('source', 'N/A')}]"
        
        content = doc.page_content
        
        # 주차 정보가 있으면 추가
        if 'week' in meta:
            header += f"\n주차: {meta['week']}"
            
        formatted_strings.append(f"{header}\n{content}")
    
    return "\n\n---\n\n".join(formatted_strings)

def _extract_week_from_query(query: str) -> Tuple[str, int | None]:
    """
    질문에서 '3주차', 'week 5' 등의 표현을 찾아 (정제된 질문, 주차 숫자)를 반환합니다.
    """
    match = WEEK_NUM_REGEX.search(query)
    week_num = None
    cleaned_query = query
    
    if match:
        week_num = int(match.group(1))
        # 질문에서 주차 관련 표현 제거 (더 나은 검색을 위해)
        # 예: "3주차 실험 내용" -> "실험 내용"
        cleaned_query = WEEK_NUM_REGEX.sub("", query).strip()
        
    return cleaned_query, week_num

def _expand_textbook_query(query: str) -> str:
    """
    질문에 '교재', 'textbook' 등이 포함되면 동의어를 추가하여 검색 성능을 높입니다.
    """
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in TEXTBOOK_KEYWORDS):
        # 원본 질문 + 키워드 동의어 결합
        expanded_query = query + " " + " ".join(TEXTBOOK_KEYWORDS)
        return expanded_query
    return query

def filtered_search(query_str: str, retriever: Chroma) -> List[Document]: # [수정] input_dict -> query_str
    """
    사용자 질문(str)을 받아 필터링된 검색을 수행합니다.
    - 주차 필터링
    - 교재 키워드 확장
    """
    query = query_str # [수정] query_str을 query로 받음
    
    # 1. 주차 정보 추출
    cleaned_query, week_num = _extract_week_from_query(query)
    
    # 2. 교재 키워드 확장
    search_query = _expand_textbook_query(cleaned_query)

    # 3. 검색 수행
    search_kwargs = {"k": 5} # 상위 5개 검색
    
    if week_num is not None:
        # 'week' 메타데이터 필터링 추가
        search_kwargs["filter"] = {"week": week_num}
        print(f"  [Debug] 주차 필터 검색 (주차: {week_num}, 쿼리: '{search_query}')")
    else:
        print(f"  [Debug] 일반 검색 (쿼리: '{search_query}')")

    # [수정] Chroma 객체의 similarity_search를 직접 사용
    return retriever.similarity_search(
        search_query, 
        k=search_kwargs.get("k", 5),
        filter=search_kwargs.get("filter")
    )


def build_rag_chain(vectordb: Chroma) -> Runnable:
    """
    LangChain Expression Language (LCEL)를 사용하여 RAG 체인을 구성합니다.
    [수정] main.py에서 str을 입력받도록 체인 구조 변경
    """
    
    # 1. LLM 모델 정의 (gpt-4o-mini 사용)
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0 # 일관성 있는 답변을 위해 0.0
    )
    
    # 2. 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "질문: {question}")
    ])
    
    # 3. 리트리버 정의 (Chroma 객체 자체)
    retriever = vectordb

    # 4. LCEL 체인 구성 (str -> dict 변환 포함)
    
    # 4-1. 메인 RAG 답변 생성 체인
    rag_chain_internal = (
        RunnableParallel(
            # "context" 키 생성: 
            # 1. 입력된 딕셔너리에서 "question" 키의 값을 filtered_search로 전달
            # 2. filtered_search의 결과를 _format_docs_with_metadata 함수로 전달
            context=(lambda x: x["question"]) | RunnableLambda(lambda q: filtered_search(q, retriever)) | RunnableLambda(_format_docs_with_metadata),
            
            # "question" 키 생성:
            # 1. 입력된 딕셔너리에서 "question" 키의 값을 그대로 통과
            question=(lambda x: x["question"])
        )
        | prompt         # 프롬프트 템플릿에 context와 question 삽입
        | llm            # LLM에 프롬프트 전달
        | StrOutputParser() # LLM의 응답(AIMessage)을 문자열로 파싱
    )
    
    # 4-2. 검색된 문서를 함께 반환하는 체인 (출처 표시용)
    chain_with_source = RunnableParallel(
        # "answer" 키: 위에서 정의한 rag_chain 실행
        answer=rag_chain_internal,
        # "source_docs" 키: 
        # 1. 입력된 딕셔너리에서 "question" 키의 값을 filtered_search로 전달
        source_docs=(lambda x: x["question"]) | RunnableLambda(lambda q: filtered_search(q, retriever))
    )
    
    # 4-3. (최종) str -> dict 변환기
    # main.py에서 전달받은 query(str)를 {"question": query} 딕셔너리로 변환하여
    # chain_with_source로 전달하는 최종 체인을 구성
    final_chain = (lambda query_str: {"question": query_str}) | chain_with_source
    
    return final_chain

