import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, ChevronRight, ChevronDown } from 'lucide-react';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

interface PreviewPanelProps {
  content: string;
  isActive: boolean;
  isComplete: boolean;
}

export function PreviewPanel({ content, isActive, isComplete }: PreviewPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!isActive && !isComplete) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ x: 400, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        exit={{ x: 400, opacity: 0 }}
        transition={{ type: 'spring', damping: 25, stiffness: 200 }}
        className="fixed right-4 top-24 w-80 z-40"
      >
        <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg shadow-xl border border-blue-200 overflow-hidden">
          {/* Header */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full px-4 py-3 bg-gradient-to-r from-blue-500 to-indigo-500 text-white flex items-center justify-between hover:from-blue-600 hover:to-indigo-600 transition-all"
          >
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              <span className="font-semibold text-sm">
                {isComplete ? '참고한 배경 지식' : '잠깐만, 이것도 알아두면 좋아!'}
              </span>
            </div>
            <motion.div
              animate={{ rotate: isExpanded ? 0 : -90 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronDown className="w-4 h-4" />
            </motion.div>
          </button>

          {/* Content */}
          <AnimatePresence>
            {isExpanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="overflow-hidden"
              >
                <div className="p-4">
                  {!isComplete && (
                    <div className="mb-3 flex items-center gap-2 text-xs text-blue-600 font-medium">
                      <div className="flex gap-1">
                        <motion.span
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{ duration: 1.5, repeat: Infinity }}
                          className="w-1.5 h-1.5 bg-blue-500 rounded-full"
                        />
                        <motion.span
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{ duration: 1.5, repeat: Infinity, delay: 0.2 }}
                          className="w-1.5 h-1.5 bg-blue-500 rounded-full"
                        />
                        <motion.span
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{ duration: 1.5, repeat: Infinity, delay: 0.4 }}
                          className="w-1.5 h-1.5 bg-blue-500 rounded-full"
                        />
                      </div>
                      <span>정확한 답변 찾는 중...</span>
                    </div>
                  )}

                  <div className="prose prose-sm prose-blue max-w-none">
                    <div className="text-sm text-gray-700 leading-relaxed">
                      <ReactMarkdown>{content}</ReactMarkdown>
                    </div>
                  </div>

                  {!isComplete && (
                    <div className="mt-3 pt-3 border-t border-blue-100 text-xs text-gray-500">
                      이 내용은 일반적인 배경 지식이야. 곧 정확한 강의 정보를 알려줄게!
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Collapsed State Indicator */}
          {!isExpanded && (
            <div className="px-4 py-2 text-xs text-gray-500 text-center">
              클릭해서 다시 보기
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

