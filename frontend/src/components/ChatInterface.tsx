import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Message as MessageType } from '../types';
import { Message } from './Message';
import { MessageInput } from './MessageInput';
import { BookOpen, Briefcase, X } from 'lucide-react';
import { jobRecommendations, JobRecommendation } from '../data/jobRecommendations';
import { JobRecommendationCard } from './JobRecommendationCard';

interface ChatInterfaceProps {
  messages: MessageType[];
  onSendMessage: (message: string) => void;
  isProcessing: boolean;
}

export function ChatInterface({ messages, onSendMessage, isProcessing }: ChatInterfaceProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [showJobRecommendations, setShowJobRecommendations] = useState(false);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleJobSelect = (job: JobRecommendation) => {
    // 기술 스택과 역량을 기반으로 쿼리 생성
    const skills = job.skills.join(', ');
    const requirements = job.requirements.slice(0, 3).join(', ');
    
    const query = `${skills} 기술을 배울 수 있고, ${requirements} 등의 역량을 키울 수 있는 강의를 추천해줘. ${job.position} 직무를 준비하려고 해.`;
    
    onSendMessage(query);
    setShowJobRecommendations(false);
  };

  return (
    <div className="relative h-full flex flex-col">
      {/* Messages Container - 스크롤 가능 */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto px-4 py-6 pb-32"
        style={{ scrollBehavior: 'smooth' }}
      >
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <AnimatePresence>
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
          </AnimatePresence>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - 플로팅 하단 고정 */}
      <div className="fixed bottom-0 left-0 right-0 z-50 border-t border-gray-200 bg-white/95 backdrop-blur-sm shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="max-w-4xl mx-auto">
            <MessageInput
              onSend={onSendMessage}
              disabled={isProcessing}
              placeholder={
                isProcessing 
                  ? "답변 생성 중..." 
                  : "질문을 입력하세요..."
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  const exampleQueries = [
    "자연언어처리 강의의 3주차 내용이 뭐야?",
    "자연언어처리 강의의 평가 방법은?",
    "자연언어처리 강의의 교재는 무엇인가요?",
    "AI융합학부 2학년 강의 중에 어떤 것들이 있어?",
  ];

  const handleJobSelect = (job: JobRecommendation) => {
    // 기술 스택과 역량을 기반으로 쿼리 생성
    const skills = job.skills.join(', ');
    const requirements = job.requirements.slice(0, 3).join(', ');
    
    const query = `${skills} 기술을 배울 수 있고, ${requirements} 등의 역량을 키울 수 있는 강의를 추천해줘. ${job.position} 직무를 준비하려고 해.`;
    
    const event = new CustomEvent('example-query', { detail: query });
    window.dispatchEvent(event);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center text-center px-4 py-8 min-h-full"
    >
      <div className="mb-6 mt-8">
        <div className="w-16 h-16 bg-gradient-to-br from-slate-700 to-slate-800 rounded-2xl flex items-center justify-center mb-4 mx-auto shadow-md">
          <BookOpen className="w-8 h-8 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          강의계획서 검색
        </h2>
        <p className="text-gray-600">
          강의계획서에 대해 궁금한 것을 물어보세요
        </p>
      </div>

      <div className="w-full max-w-2xl">
        <p className="text-sm font-semibold text-gray-700 mb-3">
          예시 질문
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {exampleQueries.map((query, index) => (
            <motion.button
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ y: -2 }}
              className="text-left p-4 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:shadow-sm transition-all"
              onClick={() => {
                const event = new CustomEvent('example-query', { detail: query });
                window.dispatchEvent(event);
              }}
            >
              <p className="text-sm text-gray-700">{query}</p>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Job Recommendations Section */}
      <div className="mt-6 w-full max-w-6xl">
        <div className="mb-3 flex items-center gap-2 text-gray-700">
          <Briefcase className="w-5 h-5" />
          <h3 className="font-semibold">직무별 추천</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 mb-6">
          {jobRecommendations.map((job, index) => (
            <JobRecommendationCard
              key={job.id}
              job={job}
              onSelect={handleJobSelect}
              index={index}
            />
          ))}
        </div>
      </div>

      
    </motion.div>
  );
}

