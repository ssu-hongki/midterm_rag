# 강의계획서 RAG 챗봇 (React + FastAPI)

에이전틱 UX를 갖춘 강의계획서 검색 챗봇입니다. React 프론트엔드와 FastAPI 백엔드로 구성되어 있습니다.

## ✨ 주요 특징

### 🎯 에이전틱 UX
- **실시간 진행 상황 표시**: AI가 무엇을 하고 있는지 단계별로 보여줍니다
- **투명한 처리 과정**: 질문 변환, 필터링, 검색, 재랭킹 등 모든 단계를 시각화
- **스트리밍 답변**: 답변이 생성되는 대로 실시간으로 표시
- **신뢰감 있는 디자인**: 부드러운 애니메이션과 명확한 피드백

### 🔍 강력한 검색 기능
- **자동 질문 변환**: 오타나 모호한 표현을 자동으로 수정
- **질문 확장**: 다양한 관점에서 검색하여 더 나은 결과 제공
- **메타데이터 필터링**: 학과, 학년, 강좌명 등으로 자동 필터링
- **재랭킹**: Cross-encoder를 사용한 정확한 결과 재정렬

### 💬 채팅 UX
- 카카오톡 스타일의 직관적인 채팅 인터페이스
- 메시지 기록 유지
- 컨텍스트 정보 확장/축소
- 반응형 디자인

## 📁 프로젝트 구조

```
midterm_rag/
├── backend/              # FastAPI 백엔드
│   ├── main.py          # API 서버
│   └── requirements.txt # Python 의존성
├── frontend/            # React 프론트엔드
│   ├── src/
│   │   ├── components/  # React 컴포넌트
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── Message.tsx
│   │   │   ├── MessageInput.tsx
│   │   │   ├── ProgressIndicator.tsx
│   │   │   └── ContextCard.tsx
│   │   ├── App.tsx      # 메인 앱
│   │   ├── api.ts       # API 클라이언트
│   │   └── types.ts     # TypeScript 타입
│   ├── package.json
│   └── vite.config.ts
├── data/                # 데이터 디렉토리
├── rag_chain.py         # RAG 로직
└── vector_store.py      # 벡터 스토어
```

## 🚀 시작하기

### 사전 요구사항

- Python 3.11+
- Node.js 18+
- OpenAI API Key

### 1. 환경 설정

프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키를 설정합니다:

```bash
OPENAI_API_KEY=your_api_key_here
```

### 2. 백엔드 설정 및 실행

```bash
# 백엔드 의존성 설치
cd backend
pip install -r requirements.txt

# 백엔드 서버 실행
python main.py
```

백엔드 서버가 `http://localhost:8000`에서 실행됩니다.

### 3. 프론트엔드 설정 및 실행

새 터미널을 열고:

```bash
# 프론트엔드 의존성 설치
cd frontend
npm install

# 개발 서버 실행
npm run dev
```

프론트엔드가 `http://localhost:5173`에서 실행됩니다.

### 4. 브라우저에서 접속

브라우저에서 `http://localhost:5173`을 열면 챗봇을 사용할 수 있습니다.

## 🎨 주요 기능 설명

### 1. 실시간 진행 상황 표시

질문을 입력하면 AI가 다음 단계를 실시간으로 보여줍니다:

1. **질문 분석** 🔍
   - 질문을 명확하게 변환
   - 오타 및 문법 오류 수정

2. **강의 필터링** 🏷️
   - 메타데이터 필터 추출
   - 관련 강의만 선택

3. **문서 검색** 📚
   - 질문 확장 (여러 관점에서 검색)
   - 벡터 유사도 검색

4. **결과 재정렬** 🎯
   - Cross-encoder를 사용한 재랭킹
   - 가장 관련성 높은 결과 선택

5. **답변 생성** ✨
   - GPT-4o-mini를 사용한 답변 생성
   - 스트리밍 방식으로 실시간 표시

### 2. 투명한 정보 표시

- **질문 변환 정보**: 원본 질문과 변환된 질문을 나란히 표시
- **적용된 필터**: 자동으로 추출된 메타데이터 필터 표시
- **참고한 문서**: 답변에 사용된 강의계획서 정보를 카드 형태로 제공
  - 강좌명, 과목코드, 담당교수
  - 유사도 점수
  - 내용 미리보기 (확장 가능)

### 3. 설정 옵션

헤더의 설정 버튼을 클릭하여:

- **재랭킹 사용**: Cross-encoder 재랭킹 활성화/비활성화
- **질문 변환/확장**: 자동 질문 변환 및 확장 활성화/비활성화

## 🎯 예시 질문

- "자연언어처리 강의의 3주차 내용이 뭐야?"
- "자연언어처리 강의의 평가 방법은?"
- "자연언어처리 강의의 교재는 무엇인가요?"
- "AI융합학부 2학년 강의 중에 어떤 것들이 있어?"

## 📚 API 엔드포인트

### `GET /health`
서버 상태 확인

### `POST /api/config`
RAG 설정 업데이트

### `POST /api/query/stream`
스트리밍 방식으로 질문 처리 (Server-Sent Events)

### `POST /api/query`
일반 방식으로 질문 처리

## 🛠️ 개발

### 프론트엔드 개발

```bash
cd frontend
npm run dev      # 개발 서버
npm run build    # 프로덕션 빌드
npm run preview  # 빌드 결과 미리보기
```

### 백엔드 개발

```bash
cd backend
# 코드 수정 후 서버 재시작
python main.py
```

FastAPI는 자동 리로딩을 지원하므로 개발 중에는:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🎨 기술 스택

### 프론트엔드
- **React 18** - UI 프레임워크
- **TypeScript** - 타입 안정성
- **Vite** - 빌드 도구
- **Tailwind CSS** - 스타일링
- **Framer Motion** - 애니메이션
- **React Markdown** - 마크다운 렌더링
- **Lucide React** - 아이콘

### 백엔드
- **FastAPI** - API 프레임워크
- **OpenAI API** - LLM 및 임베딩
- **Sentence Transformers** - Cross-encoder 재랭킹
- **NumPy** - 벡터 연산

## 🐛 문제 해결

### 백엔드 연결 실패
1. 백엔드 서버가 실행 중인지 확인
2. `.env` 파일에 `OPENAI_API_KEY`가 설정되어 있는지 확인
3. 포트 8000이 이미 사용 중인지 확인

### 프론트엔드 빌드 오류
1. `node_modules` 삭제 후 재설치: `rm -rf node_modules && npm install`
2. Node.js 버전 확인: `node -v` (18 이상)

### 벡터 스토어 없음
1. 먼저 `preprocess_pdfs.py`를 실행하여 PDF 전처리
2. 그 다음 `rebuild_vector_store.py` 실행하여 벡터 스토어 구축

## 📝 라이선스

이 프로젝트는 교육 목적으로 만들어졌습니다.

## 🙏 감사의 말

- OpenAI API
- Sentence Transformers
- React 커뮤니티
- FastAPI 커뮤니티

