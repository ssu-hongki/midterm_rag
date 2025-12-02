# 🚀 빠른 시작 가이드

## 1. 사전 준비

### 필수 요구사항 확인
- ✅ Python 3.11 이상
- ✅ Node.js 18 이상
- ✅ OpenAI API Key

### API 키 설정
프로젝트 루트에 `.env` 파일 생성:

```bash
OPENAI_API_KEY=your_api_key_here
```

## 2. 의존성 설치

### 백엔드 의존성 설치
```bash
# 가상환경이 없으면 활성화
source venv/bin/activate

# 백엔드 의존성 설치
pip install -r backend/requirements.txt
```

### 프론트엔드 의존성 설치
```bash
cd frontend
npm install
cd ..
```

## 3. 벡터 스토어 구축 (최초 1회)

```bash
# 1. PDF 전처리
python preprocess_pdfs.py

# 2. 벡터 스토어 구축
python rebuild_vector_store.py
```

## 4. 앱 실행

### 방법 1: 통합 실행 스크립트 (추천) 🎯

```bash
./start_app.sh
```

백엔드와 프론트엔드가 동시에 실행됩니다!
- 백엔드: http://localhost:8000
- 프론트엔드: http://localhost:5173

### 방법 2: 수동 실행

**터미널 1 - 백엔드:**
```bash
./start_backend.sh
```

**터미널 2 - 프론트엔드:**
```bash
./start_frontend.sh
```

## 5. 브라우저에서 접속

브라우저를 열고 http://localhost:5173 으로 이동하세요!

## 💡 사용 예시

챗봇에 다음과 같은 질문을 해보세요:

1. "자연언어처리 강의의 3주차 내용이 뭐야?"
2. "자연언어처리 강의의 평가 방법은?"
3. "AI융합학부 2학년 강의 중에 어떤 것들이 있어?"

## 🎨 에이전틱 UX 특징

실행하면 다음과 같은 기능을 경험할 수 있습니다:

✨ **실시간 진행 상황**
- 🔍 질문 분석 중...
- 🏷️ 강의 필터링 중...
- 📚 문서 검색 중...
- 🎯 결과 재정렬 중...
- ✨ 답변 생성 중...

✨ **투명한 정보 표시**
- 질문이 어떻게 변환되었는지 표시
- 어떤 필터가 적용되었는지 표시
- 어떤 강의계획서를 참고했는지 표시

✨ **스트리밍 답변**
- 답변이 생성되는 대로 실시간으로 표시
- 기다리는 시간이 지루하지 않음

## 🛠️ 문제 해결

### "연결 실패" 표시될 때
1. 백엔드 서버가 실행 중인지 확인
2. 터미널에서 에러 메시지 확인

### 벡터 스토어 오류
```bash
# 벡터 스토어 재구축
python rebuild_vector_store.py
```

### 포트 충돌
이미 포트를 사용 중이라면:
```bash
# 백엔드: backend/main.py의 포트 변경
# 프론트엔드: frontend/vite.config.ts의 포트 변경
```

## 📚 더 자세한 정보

전체 문서는 `README_REACT.md`를 참고하세요.

