# 강의계획서 RAG 챗봇 🎓

숭실대학교 강의계획서 PDF를 기반으로 한 검색 증강 생성(RAG) 챗봇입니다. OpenAI의 임베딩 모델과 GPT-4를 활용하여 강의 관련 질문에 정확하게 답변합니다.

## 주요 기능 ✨

### 1. 고급 검색 기능
- **하이브리드 검색**: 벡터 검색(Semantic Search)과 BM25(Keyword Search)를 결합
- **Cross-Encoder 재랭킹**: 검색 결과를 더욱 정확하게 재정렬
- **질문 변환 및 확장**: 모호한 질문을 명확하게 변환하고 다양한 관점에서 확장
- **메타데이터 필터링**: 학과, 학년, 강좌명 등으로 자동 필터링

### 2. 지능형 질의 처리
- 오타 자동 수정
- 모호한 표현 명확화
- 다양한 질문 형식 지원 (예: "nlp 3주차 뭐함?" → "자연언어처리 3주차 실습 내용은?")

### 3. 사용자 친화적 인터페이스
- Streamlit 기반 웹 UI
- 검색 과정 시각화 (질문 변환, 필터 적용 확인)
- 참고 문서 출처 표시

## 프로젝트 구조 📁

```
midterm_rag/
├── data/
│   ├── pdfs/                          # 원본 PDF 파일
│   ├── processed/                     # 전처리된 JSON 데이터
│   │   └── course_chunks.json
│   └── vector_store/                  # 벡터 저장소
│       ├── embeddings.npy
│       └── metadatas.json
├── preprocess_pdfs.py                 # PDF 전처리 스크립트
├── data_loader.py                     # 데이터 로딩 유틸리티
├── vector_store.py                    # 벡터 저장소 관리
├── rag_chain.py                       # RAG 체인 구현
├── utils.py                           # 유틸리티 함수
├── streamlit_app.py                   # Streamlit 웹 앱
├── main_rag.py                        # CLI 인터페이스
├── rebuild_vector_store.py            # 벡터 저장소 재구축
├── requirements.txt                   # 의존성 패키지
├── .env                               # 환경 변수 (API 키)
└── README.md
```

## 설치 방법 🚀

### 1. 저장소 클론
```bash
git clone <repository-url>
cd midterm_rag
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 설정합니다:
```env
OPENAI_API_KEY=your-api-key-here
```

## 사용 방법 📖

### 1. PDF 전처리
강의계획서 PDF 파일을 `data/pdfs/` 디렉토리에 넣고 전처리를 실행합니다:
```bash
python preprocess_pdfs.py
```

### 2. 벡터 저장소 구축
전처리된 데이터로 벡터 저장소를 구축합니다:
```bash
python rebuild_vector_store.py
```

### 3. Streamlit 웹 앱 실행
```bash
streamlit run streamlit_app.py
```

브라우저에서 자동으로 열리며, 다음 주소로 접속할 수 있습니다:
- Local: http://localhost:8501
- Network: http://<your-ip>:8501

### 4. CLI 인터페이스 사용 (선택사항)
터미널에서 직접 질문할 수 있습니다:
```bash
python main_rag.py
```

## 사용 예시 💡

### 질문 예시
```
✅ 좋은 질문:
- "자연언어처리 강의의 주요 교재는?"
- "데이터베이스 3주차 실습 내용이 뭐야?"
- "AI융합학부 3학년 학생이 들을 수 있는 강의는?"

✅ 모호한 질문도 자동 처리:
- "nlp 3주차 뭐함?" → "자연언어처리 3주차 실습 내용은?"
- "db 교재" → "데이터베이스 강의의 주요 교재는?"
```

### 검색 설정
Streamlit 사이드바에서 다음 옵션을 조정할 수 있습니다:
- **재랭킹 사용**: Cross-encoder를 사용하여 검색 정확도 향상 (권장)
- **질문 변환/확장 사용**: 질문을 자동으로 개선하고 확장 (권장)

## 기술 스택 🛠️

### Core
- **Python 3.8+**
- **OpenAI API**: GPT-4 & text-embedding-3-small
- **Streamlit**: 웹 인터페이스

### 검색 & 랭킹
- **sentence-transformers**: Cross-encoder 재랭킹
- **rank-bm25**: BM25 키워드 검색
- **NumPy**: 벡터 연산

### PDF 처리
- **pdfplumber**: PDF 테이블 추출

## 주요 알고리즘 🧮

### 1. 하이브리드 검색
```python
hybrid_score = α × vector_score + (1-α) × bm25_score
```
- 기본 α = 0.5 (벡터와 BM25 점수를 동등하게 반영)

### 2. 재랭킹
```python
final_score = 0.1 × hybrid_score + 0.9 × rerank_score
```
- Cross-encoder의 재랭킹 점수를 강하게 반영

### 3. 질문 확장
- GPT-4를 사용하여 원본 질문을 2-3개의 관련 질문으로 확장
- 각 확장 질문으로 검색 후 결과를 병합

## 성능 최적화 팁 ⚡

1. **재랭킹 사용**: 정확도를 크게 향상시키지만 속도가 약간 느려집니다
2. **질문 확장 제한**: 너무 많은 확장은 노이즈를 증가시킬 수 있습니다
3. **메타데이터 필터**: 가능한 경우 학과, 학년 등을 명시하면 더 정확합니다

## 문제 해결 🔧

### 벡터 저장소 오류
```bash
# 벡터 저장소 재구축
python rebuild_vector_store.py
```

### PDF 전처리 실패
- PDF 파일이 `data/pdfs/` 디렉토리에 있는지 확인
- PDF 파일이 올바른 강의계획서 형식인지 확인

### API 키 오류
- `.env` 파일이 프로젝트 루트에 있는지 확인
- `OPENAI_API_KEY`가 올바르게 설정되었는지 확인
