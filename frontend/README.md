# 강의계획서 RAG 챗봇 - 프론트엔드

React + TypeScript + Vite로 구축된 에이전틱 UX 챗봇 프론트엔드입니다.

## 기술 스택

- **React 18** - UI 프레임워크
- **TypeScript** - 타입 안정성
- **Vite** - 빠른 빌드 도구
- **Tailwind CSS** - 유틸리티 기반 스타일링
- **Framer Motion** - 부드러운 애니메이션
- **React Markdown** - 마크다운 렌더링
- **Lucide React** - 아름다운 아이콘

## 설치 및 실행

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 결과 미리보기
npm run preview
```

## 컴포넌트 구조

```
src/
├── components/
│   ├── ChatInterface.tsx      # 메인 채팅 인터페이스
│   ├── Message.tsx             # 개별 메시지 컴포넌트
│   ├── MessageInput.tsx        # 메시지 입력 컴포넌트
│   ├── ProgressIndicator.tsx  # 진행 상황 표시
│   └── ContextCard.tsx         # 컨텍스트 정보 카드
├── App.tsx                     # 메인 앱
├── api.ts                      # API 클라이언트
├── types.ts                    # TypeScript 타입 정의
├── main.tsx                    # 엔트리 포인트
└── index.css                   # 글로벌 스타일
```

## 주요 기능

### 1. 실시간 스트리밍
Server-Sent Events (SSE)를 사용하여 백엔드로부터 실시간으로 데이터를 받아 표시합니다.

### 2. 진행 상황 시각화
질문 처리 단계를 실시간으로 시각화하여 사용자에게 투명성을 제공합니다.

### 3. 부드러운 애니메이션
Framer Motion을 사용하여 모든 상태 전환에 부드러운 애니메이션을 적용합니다.

### 4. 반응형 디자인
모바일부터 데스크톱까지 모든 화면 크기에 대응합니다.

## 환경 변수

`.env` 파일에서 설정 가능:

```bash
VITE_API_URL=http://localhost:8000
```

## 개발 가이드

### 새 컴포넌트 추가
1. `src/components/` 디렉토리에 `.tsx` 파일 생성
2. 타입은 `src/types.ts`에 정의
3. 스타일은 Tailwind CSS 클래스 사용

### 상태 관리
현재는 React의 내장 상태 관리를 사용합니다. 필요시 Zustand나 Redux 추가 가능합니다.

### API 호출
`src/api.ts`의 `RAGAPIClient` 클래스를 사용합니다.

## 빌드 최적화

- 코드 스플리팅 자동 적용
- Tree shaking으로 불필요한 코드 제거
- 프로덕션 빌드 시 자동 압축

## 라이선스

교육 목적

