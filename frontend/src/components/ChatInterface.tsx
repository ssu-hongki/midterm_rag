import { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Message as MessageType } from '../types';
import { Message } from './Message';
import { MessageInput } from './MessageInput';
import { BookOpen } from 'lucide-react';

interface ChatInterfaceProps {
  messages: MessageType[];
  onSendMessage: (message: string) => void;
  isProcessing: boolean;
}

export function ChatInterface({ messages, onSendMessage, isProcessing }: ChatInterfaceProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      {/* Messages Container */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto px-4 py-6"
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

      {/* Input Area */}
      <div className="flex-shrink-0 border-t border-gray-200 bg-gradient-to-b from-transparent to-gray-50 px-4 py-4">
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
  );
}

function EmptyState() {
  const exampleQueries = [
    "자연언어처리 강의의 3주차 내용이 뭐야?",
    "자연언어처리 강의의 평가 방법은?",
    "자연언어처리 강의의 교재는 무엇인가요?",
    "AI융합학부 2학년 강의 중에 어떤 것들이 있어?",
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center justify-center h-full text-center px-4"
    >
      <div className="mb-8">
        <div className="w-20 h-20 bg-gradient-to-br from-primary-400 to-primary-600 rounded-3xl flex items-center justify-center mb-4 mx-auto shadow-lg">
          <BookOpen className="w-10 h-10 text-white" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          강의계획서 RAG 챗봇
        </h2>
        <p className="text-gray-600">
          강의계획서에 대해 궁금한 것을 물어보세요
        </p>
      </div>

      <div className="w-full max-w-2xl">
        <p className="text-sm font-medium text-gray-700 mb-3">
          💡 이런 질문을 해보세요
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {exampleQueries.map((query, index) => (
            <motion.button
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="text-left p-4 bg-white border-2 border-gray-200 rounded-xl hover:border-primary-300 hover:shadow-md transition-all"
              onClick={() => {
                // This could trigger the query
                const event = new CustomEvent('example-query', { detail: query });
                window.dispatchEvent(event);
              }}
            >
              <p className="text-sm text-gray-700">{query}</p>
            </motion.button>
          ))}
        </div>
      </div>

      <div className="mt-12 p-4 bg-blue-50 border border-blue-200 rounded-xl max-w-2xl">
        <p className="text-sm text-blue-900">
          <span className="font-semibold">✨ AI가 자동으로:</span><br />
          • 질문을 분석하고 명확하게 변환합니다<br />
          • 관련된 강의를 필터링합니다<br />
          • 가장 관련성 높은 정보를 찾아줍니다<br />
          • 처리 과정을 실시간으로 보여줍니다<br />
          • 💬 이전 대화를 기억하고 맥락을 이해합니다
        </p>
      </div>
    </motion.div>
  );
}

