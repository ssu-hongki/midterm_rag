// 채용 공고 기반 직무 추천 데이터

export interface JobRecommendation {
  id: string;
  company: string;
  position: string;
  category: string;
  skills: string[];
  requirements: string[];
  preferredSkills: string[];
  color: string;
  url: string;
}

export const jobRecommendations: JobRecommendation[] = [
  {
    id: 'miridi-backend',
    company: '미리디',
    position: '미리캔버스 백엔드 개발자',
    category: '백엔드',
    skills: ['Java', 'Spring', 'AWS', 'MSA'],
    requirements: [
      'RESTful API 설계 및 개발 경험',
      'DB 쿼리 최적화 경험',
      '대용량 트래픽 최적화 경험',
    ],
    preferredSkills: [
      'MSA 전환 경험',
      '실시간 저장 및 동시 편집 구현',
      'AI 추천 시스템 개발',
    ],
    color: 'from-slate-600 to-slate-700',
    url: 'https://www.wanted.co.kr/wd/317108',
  },
  {
    id: 'kakao-healthcare-frontend',
    company: '카카오헬스케어',
    position: '파스타 프론트엔드 개발자',
    category: '프론트엔드',
    skills: ['React', 'TypeScript', 'Next.js', 'JavaScript'],
    requirements: [
      '프론트엔드 개발 경험',
      'React 기반 SPA 개발 경험',
      'TypeScript 사용 경험',
    ],
    preferredSkills: [
      '헬스케어 서비스 개발 경험',
      '사용자 경험 최적화',
      '컴포넌트 설계 및 개발',
    ],
    color: 'from-blue-600 to-blue-700',
    url: 'https://www.wanted.co.kr/wd/315257',
  },
  {
    id: 'general-fullstack',
    company: '종합',
    position: '풀스택 개발자',
    category: '풀스택',
    skills: ['React', 'Node.js', 'TypeScript', 'Database'],
    requirements: [
      '프론트엔드 및 백엔드 개발 경험',
      'RESTful API 설계 및 개발',
      '데이터베이스 설계 경험',
    ],
    preferredSkills: [
      '클라우드 서비스 경험',
      'CI/CD 구축 경험',
      'Docker/K8s 경험',
    ],
    color: 'from-slate-600 to-slate-700',
    url: '#',
  },
  {
    id: 'ai-engineer',
    company: '종합',
    position: 'AI/ML 엔지니어',
    category: 'AI/ML',
    skills: ['Python', 'TensorFlow', 'PyTorch', 'Machine Learning'],
    requirements: [
      '머신러닝/딥러닝 모델 개발 경험',
      'Python 기반 개발 경험',
      '데이터 전처리 및 분석 경험',
    ],
    preferredSkills: [
      'LLM 모델 파인튜닝',
      'RAG 시스템 구축',
      'MLOps 경험',
    ],
    color: 'from-indigo-600 to-indigo-700',
    url: '#',
  },
  {
    id: 'data-engineer',
    company: '종합',
    position: '데이터 엔지니어',
    category: '데이터',
    skills: ['Python', 'SQL', 'Spark', 'Kafka'],
    requirements: [
      '데이터 파이프라인 구축 경험',
      'SQL 및 NoSQL 데이터베이스 경험',
      '대용량 데이터 처리 경험',
    ],
    preferredSkills: [
      '데이터 웨어하우스 구축',
      '실시간 데이터 처리',
      'ETL 프로세스 설계',
    ],
    color: 'from-blue-600 to-blue-700',
    url: '#',
  },
  {
    id: 'devops-engineer',
    company: '종합',
    position: 'DevOps 엔지니어',
    category: 'DevOps',
    skills: ['Docker', 'Kubernetes', 'AWS', 'CI/CD'],
    requirements: [
      '클라우드 인프라 구축 경험',
      'CI/CD 파이프라인 구축',
      '컨테이너 오케스트레이션 경험',
    ],
    preferredSkills: [
      'IaC (Terraform, CloudFormation)',
      '모니터링 시스템 구축',
      '보안 및 컴플라이언스',
    ],
    color: 'from-slate-700 to-slate-800',
    url: '#',
  },
  {
    id: 'mobile-developer',
    company: '종합',
    position: '모바일 개발자',
    category: '모바일',
    skills: ['React Native', 'Flutter', 'iOS', 'Android'],
    requirements: [
      '모바일 앱 개발 경험',
      'iOS 또는 Android 개발 경험',
      'REST API 연동 경험',
    ],
    preferredSkills: [
      '크로스 플랫폼 개발 경험',
      '앱 성능 최적화',
      '푸시 알림 구현',
    ],
    color: 'from-blue-600 to-blue-700',
    url: '#',
  },
  {
    id: 'security-engineer',
    company: '종합',
    position: '보안 엔지니어',
    category: '보안',
    skills: ['Network Security', 'OWASP', 'Penetration Testing', 'Security'],
    requirements: [
      '보안 취약점 분석 경험',
      '보안 정책 수립 경험',
      '침투 테스트 경험',
    ],
    preferredSkills: [
      'CISSP, CEH 등 자격증',
      '클라우드 보안',
      'Zero Trust 아키텍처',
    ],
    color: 'from-gray-700 to-gray-800',
    url: '#',
  },
  {
    id: 'blockchain-developer',
    company: '종합',
    position: '블록체인 개발자',
    category: '블록체인',
    skills: ['Solidity', 'Web3', 'Ethereum', 'Smart Contract'],
    requirements: [
      '스마트 컨트랙트 개발 경험',
      'Solidity 개발 경험',
      '블록체인 기술 이해',
    ],
    preferredSkills: [
      'DeFi 프로토콜 개발',
      'NFT 프로젝트 경험',
      'Layer 2 솔루션',
    ],
    color: 'from-indigo-600 to-indigo-700',
    url: '#',
  },
  {
    id: 'game-developer',
    company: '종합',
    position: '게임 개발자',
    category: '게임',
    skills: ['Unity', 'Unreal Engine', 'C#', 'C++'],
    requirements: [
      '게임 엔진 사용 경험',
      '게임 로직 구현 경험',
      '3D/2D 그래픽스 이해',
    ],
    preferredSkills: [
      '멀티플레이어 게임 개발',
      '게임 최적화',
      'VR/AR 게임 개발',
    ],
    color: 'from-slate-600 to-slate-700',
    url: '#',
  },
];

