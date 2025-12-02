import { motion } from 'framer-motion';
import { Loader2, Search, Filter, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react';
import { ProgressStatus } from '../types';

interface ProgressIndicatorProps {
  progress: ProgressStatus;
  details?: string[];
}

const stepIcons = {
  analyzing: Search,
  filtering: Filter,
  searching: Search,
  reranking: Sparkles,
  generating: Loader2,
  complete: CheckCircle2,
};

const stepLabels = {
  analyzing: '1. 질문 분석',
  filtering: '2. 강의 필터링',
  searching: '3. 문서 검색',
  reranking: '4. 결과 재정렬',
  generating: '5. 답변 생성',
  complete: '완료',
};

export function ProgressIndicator({ progress, details }: ProgressIndicatorProps) {
  const Icon = stepIcons[progress.step] || Loader2;
  const isComplete = progress.step === 'complete';
  const isGenerating = progress.step === 'generating';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-4"
    >
      <div className="flex items-start gap-3">
        <div className={`
          flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
          ${isComplete ? 'bg-green-100' : 'bg-primary-100'}
        `}>
          <Icon 
            className={`
              w-5 h-5 
              ${isComplete ? 'text-green-600' : 'text-primary-600'}
              ${isGenerating ? 'animate-spin' : ''}
            `}
          />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="font-medium text-gray-900">
              {stepLabels[progress.step]}
            </h4>
            {!isComplete && (
              <div className="flex gap-1">
                <span className="typing-dot w-1.5 h-1.5 bg-primary-600 rounded-full" />
                <span className="typing-dot w-1.5 h-1.5 bg-primary-600 rounded-full" />
                <span className="typing-dot w-1.5 h-1.5 bg-primary-600 rounded-full" />
              </div>
            )}
          </div>
          
          <p className="text-sm text-gray-600">
            {progress.message}
          </p>

          {details && details.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-2 space-y-1"
            >
              {details.map((detail, index) => (
                <div 
                  key={index}
                  className="text-xs text-gray-500 flex items-center gap-2"
                >
                  <span className="w-1 h-1 bg-gray-400 rounded-full" />
                  {detail}
                </div>
              ))}
            </motion.div>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export function StepProgress({ currentStep }: { currentStep: string }) {
  const steps = ['analyzing', 'filtering', 'searching', 'reranking', 'generating'];
  const currentIndex = steps.indexOf(currentStep);

  return (
    <div className="flex items-center gap-2 mb-4">
      {steps.map((step, index) => {
        const isActive = index <= currentIndex;
        const isCurrent = index === currentIndex;
        
        return (
          <div key={step} className="flex items-center gap-2">
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ 
                scale: isCurrent ? 1.1 : 1,
                backgroundColor: isActive ? '#0ea5e9' : '#e5e7eb'
              }}
              className={`
                w-2 h-2 rounded-full
                ${isCurrent ? 'animate-pulse' : ''}
              `}
            />
            {index < steps.length - 1 && (
              <div 
                className={`
                  w-8 h-0.5 transition-colors
                  ${isActive ? 'bg-primary-500' : 'bg-gray-200'}
                `}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

