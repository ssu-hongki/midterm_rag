import { motion } from 'framer-motion';
import { BookOpen, User, Hash, FileText } from 'lucide-react';
import { Context } from '../types';
import { useState } from 'react';

interface ContextCardProps {
  context: Context;
  index: number;
}

export function ContextCard({ context, index }: ContextCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      className="bg-gradient-to-r from-gray-50 to-gray-100 border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow"
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 text-left hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="flex-shrink-0 w-6 h-6 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center text-xs font-bold">
                {index + 1}
              </span>
              <h4 className="font-medium text-gray-900 truncate">
                {context.강좌명 || '강좌명 없음'}
              </h4>
              {context.score > 0 && (
                <span className="flex-shrink-0 px-2 py-0.5 bg-green-100 text-green-700 rounded-full text-xs font-medium">
                  {(context.score * 100).toFixed(0)}%
                </span>
              )}
            </div>
            
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-600">
              {context.과목코드 && (
                <span className="flex items-center gap-1">
                  <Hash className="w-3 h-3" />
                  {context.과목코드}
                </span>
              )}
              {context.담당교수 && (
                <span className="flex items-center gap-1">
                  <User className="w-3 h-3" />
                  {context.담당교수}
                </span>
              )}
              {context.source_pdf && (
                <span className="flex items-center gap-1">
                  <FileText className="w-3 h-3" />
                  {context.source_pdf}
                </span>
              )}
            </div>
          </div>

          <motion.div
            animate={{ rotate: isExpanded ? 180 : 0 }}
            transition={{ duration: 0.2 }}
            className="flex-shrink-0 text-gray-400"
          >
            <svg 
              className="w-5 h-5" 
              fill="none" 
              viewBox="0 0 24 24" 
              stroke="currentColor"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M19 9l-7 7-7-7" 
              />
            </svg>
          </motion.div>
        </div>
      </button>

      {isExpanded && context.text_preview && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          className="px-4 pb-3 border-t border-gray-200"
        >
          <div className="pt-3">
            <p className="text-xs font-medium text-gray-700 mb-2 flex items-center gap-1">
              <BookOpen className="w-3 h-3" />
              내용 미리보기
            </p>
            <p className="text-xs text-gray-600 bg-white rounded p-2 border border-gray-200 whitespace-pre-wrap">
              {context.text_preview}
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}

