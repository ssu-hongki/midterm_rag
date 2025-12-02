#!/bin/bash

# 백엔드만 실행하는 스크립트

echo "🚀 백엔드 서버 시작..."

# 프로젝트 루트로 이동
cd "$(dirname "$0")"

# .env 파일 확인
if [ ! -f .env ]; then
    echo "❌ .env 파일이 없습니다."
    echo "프로젝트 루트에 .env 파일을 생성하고 OPENAI_API_KEY를 설정해주세요."
    exit 1
fi

# 가상환경 활성화
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "⚠️  가상환경이 없습니다. 먼저 가상환경을 생성하고 의존성을 설치해주세요:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r backend/requirements.txt"
    exit 1
fi

# 백엔드 디렉토리로 이동
cd backend

# 서버 실행
echo "✅ 백엔드 서버 실행 중..."
echo "   http://localhost:8000"
echo ""
python main.py

