#!/bin/bash

# 프론트엔드만 실행하는 스크립트

echo "🎨 프론트엔드 서버 시작..."

# 프로젝트 루트로 이동
cd "$(dirname "$0")"

# 프론트엔드 디렉토리로 이동
cd frontend

# node_modules 확인
if [ ! -d "node_modules" ]; then
    echo "⚠️  node_modules가 없습니다. 의존성을 설치합니다..."
    npm install
fi

# 개발 서버 실행
echo "✅ 프론트엔드 서버 실행 중..."
echo "   http://localhost:5173"
echo ""
npm run dev

