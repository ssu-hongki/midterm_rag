#!/bin/bash

# 강의계획서 RAG 챗봇 실행 스크립트

echo "🚀 강의계획서 RAG 챗봇 시작..."

# 터미널 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# .env 파일 확인
if [ ! -f .env ]; then
    echo -e "${RED}❌ .env 파일이 없습니다.${NC}"
    echo "프로젝트 루트에 .env 파일을 생성하고 OPENAI_API_KEY를 설정해주세요."
    exit 1
fi

# 벡터 스토어 확인
if [ ! -f "data/vector_store/embeddings.npy" ]; then
    echo -e "${YELLOW}⚠️  벡터 스토어가 없습니다.${NC}"
    echo "먼저 벡터 스토어를 구축해주세요:"
    echo "  1. python preprocess_pdfs.py"
    echo "  2. python rebuild_vector_store.py"
    exit 1
fi

# 백엔드 실행
echo -e "${BLUE}📦 백엔드 서버 시작...${NC}"
cd backend

# 백엔드 의존성 확인
if [ ! -d "../venv" ]; then
    echo -e "${YELLOW}⚠️  가상환경이 없습니다. 의존성을 설치해주세요:${NC}"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 백엔드 실행 (백그라운드)
source ../venv/bin/activate
python main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✅ 백엔드 서버 실행 (PID: $BACKEND_PID)${NC}"
echo "   http://localhost:8000"

# 프론트엔드 실행
echo -e "${BLUE}🎨 프론트엔드 서버 시작...${NC}"
cd ../frontend

# 프론트엔드 의존성 확인
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}⚠️  node_modules가 없습니다. 의존성을 설치합니다...${NC}"
    npm install
fi

# 프론트엔드 실행
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✅ 프론트엔드 서버 실행 (PID: $FRONTEND_PID)${NC}"
echo "   http://localhost:5173"

echo ""
echo -e "${GREEN}✨ 모든 서버가 시작되었습니다!${NC}"
echo ""
echo "브라우저에서 http://localhost:5173 을 열어주세요."
echo ""
echo "종료하려면 Ctrl+C를 누르세요."
echo ""

# PID 저장
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

# 종료 시그널 처리
trap cleanup INT TERM

cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 서버를 종료합니다...${NC}"
    
    if [ -f .backend.pid ]; then
        BACKEND_PID=$(cat .backend.pid)
        kill $BACKEND_PID 2>/dev/null
        rm .backend.pid
        echo -e "${GREEN}✅ 백엔드 서버 종료${NC}"
    fi
    
    if [ -f .frontend.pid ]; then
        FRONTEND_PID=$(cat .frontend.pid)
        kill $FRONTEND_PID 2>/dev/null
        rm .frontend.pid
        echo -e "${GREEN}✅ 프론트엔드 서버 종료${NC}"
    fi
    
    exit 0
}

# 대기
wait

