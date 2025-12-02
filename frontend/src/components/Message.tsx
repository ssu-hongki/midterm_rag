import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { User, Bot, AlertCircle } from 'lucide-react';
import { Message as MessageType } from '../types';
import { ProgressIndicator, StepProgress } from './ProgressIndicator';
import { ContextCard } from './ContextCard';

interface MessageProps {
  message: MessageType;
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === 'user';
  const isError = message.status === 'error';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 mb-6 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
    >
      {/* Avatar */}
      <div className={`
        flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
        ${isUser ? 'bg-primary-500' : 'bg-gray-700'}
      `}>
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <Bot className="w-5 h-5 text-white" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 ${isUser ? 'flex justify-end' : ''}`}>
        <div className={`max-w-3xl ${isUser ? 'text-right' : ''}`}>
          {/* User Message */}
          {isUser && (
            <div className={`
              inline-block px-4 py-3 rounded-2xl
              ${isUser ? 'bg-primary-500 text-white rounded-tr-sm' : 'bg-white text-gray-900 rounded-tl-sm'}
              shadow-sm
            `}>
              <p className="whitespace-pre-wrap break-words">{message.content}</p>
            </div>
          )}

          {/* Assistant Message */}
          {!isUser && (
            <div className="space-y-4">
              {/* Progress Indicator */}
              {message.status === 'processing' && message.progress && (
                <div>
                  <StepProgress currentStep={message.progress.step} />
                  <ProgressIndicator 
                    progress={message.progress}
                    details={message.expandedQueries}
                  />
                </div>
              )}

              {/* Transformed Query */}
              {message.transformedQuery && message.transformedQuery !== message.content && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="bg-blue-50 border border-blue-200 rounded-lg p-3 text-sm"
                >
                  <p className="text-blue-900 font-medium mb-1">🔍 질문 변환</p>
                  <p className="text-blue-700">
                    <span className="text-blue-500">원본:</span> {message.content}
                  </p>
                  <p className="text-blue-700">
                    <span className="text-blue-500">변환:</span> {message.transformedQuery}
                  </p>
                </motion.div>
              )}

              {/* Filters */}
              {message.filters && Object.keys(message.filters).length > 0 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="bg-purple-50 border border-purple-200 rounded-lg p-3 text-sm"
                >
                  <p className="text-purple-900 font-medium mb-2">🏷️ 적용된 필터</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(message.filters).map(([key, value]) => (
                      <span 
                        key={key}
                        className="px-2 py-1 bg-purple-100 text-purple-700 rounded-full text-xs"
                      >
                        {key}: {value}
                      </span>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Answer */}
              {message.content && (
                <div className="bg-white rounded-2xl rounded-tl-sm shadow-sm border border-gray-200 p-4">
                  {isError ? (
                    <div className="flex items-start gap-2 text-red-600">
                      <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                      <p>{message.content}</p>
                    </div>
                  ) : (
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  )}
                </div>
              )}

              {/* Typing Indicator */}
              {message.status === 'processing' && !message.content && (
                <div className="bg-white rounded-2xl rounded-tl-sm shadow-sm border border-gray-200 p-4">
                  <div className="flex gap-1.5">
                    <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                    <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                    <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
                  </div>
                </div>
              )}

              {/* Contexts */}
              {message.contexts && message.contexts.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="space-y-2"
                >
                  <p className="text-sm font-medium text-gray-700 flex items-center gap-2">
                    📚 참고한 강의계획서 ({message.contexts.length}개)
                  </p>
                  <div className="space-y-2">
                    {message.contexts.map((context, index) => (
                      <ContextCard key={index} context={context} index={index} />
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
          )}

          {/* Timestamp */}
          <p className={`
            text-xs text-gray-400 mt-1
            ${isUser ? 'text-right' : 'text-left'}
          `}>
            {message.timestamp.toLocaleTimeString('ko-KR', { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

