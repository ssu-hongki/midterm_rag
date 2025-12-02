import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Settings, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react';
import { ChatInterface } from './components/ChatInterface';
import { apiClient } from './api';
import { Message, QueryConfig, SSEEvent, ProgressStatus } from './types';

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [config, setConfig] = useState<QueryConfig>({
    use_reranking: true,
    use_query_expansion: true,
  });
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [serverStatus, setServerStatus] = useState<'checking' | 'healthy' | 'error'>('checking');

  // Check server health on mount
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    setServerStatus('checking');
    try {
      const isHealthy = await apiClient.healthCheck();
      setServerStatus(isHealthy ? 'healthy' : 'error');
    } catch (error) {
      setServerStatus('error');
    }
  };

  // Listen for example queries
  useEffect(() => {
    const handleExampleQuery = (e: Event) => {
      const customEvent = e as CustomEvent;
      if (customEvent.detail) {
        handleSendMessage(customEvent.detail);
      }
    };

    window.addEventListener('example-query', handleExampleQuery);
    return () => window.removeEventListener('example-query', handleExampleQuery);
  }, [config, isProcessing]);

  const handleSendMessage = useCallback(async (content: string) => {
    if (isProcessing) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
      status: 'complete',
    };

    setMessages(prev => [...prev, userMessage]);
    setIsProcessing(true);
    
    // 대화 히스토리 준비 (최근 메시지만, 완료된 것만)
    const chatHistory = messages
      .filter(m => m.status === 'complete' && m.content)
      .slice(-10) // 최근 10개만 (5턴)
      .map(m => ({
        role: m.role,
        content: m.content
      }));
    
    console.log(`💬 대화 히스토리: ${chatHistory.length}개 메시지 전달`);

    // Create assistant message placeholder
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      status: 'processing',
      progress: {
        step: 'analyzing',
        message: '🔍 질문 분석 중...',
      },
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      let fullAnswer = '';
      let contexts: any[] = [];
      let transformedQuery: string | undefined;
      let filters: Record<string, string> | undefined;
      let expandedQueries: string[] | undefined;
      let currentProgress: ProgressStatus = assistantMessage.progress!;

      // Stream the response (대화 히스토리 포함)
      for await (const event of apiClient.queryStream(content, config, chatHistory)) {
        setMessages(prev => prev.map(msg => {
          if (msg.id !== assistantMessageId) return msg;

          const updates: Partial<Message> = {};

          switch (event.type) {
            case 'status':
              currentProgress = {
                step: event.step as any,
                message: event.message || '',
              };
              updates.progress = currentProgress;
              break;

            case 'transformed_query':
              transformedQuery = event.transformed;
              updates.transformedQuery = transformedQuery;
              break;

            case 'filters':
              filters = event.filters;
              updates.filters = filters;
              break;

            case 'expanded_queries':
              expandedQueries = event.queries;
              updates.expandedQueries = expandedQueries;
              break;

            case 'contexts_found':
              // Just update progress
              break;

            case 'answer_start':
              currentProgress = {
                step: 'generating',
                message: '✨ 답변 생성 중...',
              };
              updates.progress = currentProgress;
              break;

            case 'answer_chunk':
              if (event.content) {
                fullAnswer += event.content;
                updates.content = fullAnswer;
              }
              break;

            case 'answer_complete':
              if (event.full_answer) {
                fullAnswer = event.full_answer;
                updates.content = fullAnswer;
              }
              break;

            case 'contexts':
              contexts = event.contexts || [];
              updates.contexts = contexts;
              break;

            case 'complete':
              updates.status = 'complete';
              updates.progress = {
                step: 'complete',
                message: '✅ 완료',
              };
              break;

            case 'error':
              updates.status = 'error';
              updates.content = event.message || '오류가 발생했습니다.';
              break;
          }

          return { ...msg, ...updates };
        }));
      }
    } catch (error) {
      console.error('Query failed:', error);
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMessageId) {
          return {
            ...msg,
            status: 'error',
            content: `오류 발생: ${error instanceof Error ? error.message : '알 수 없는 오류'}`,
          };
        }
        return msg;
      }));
    } finally {
      setIsProcessing(false);
    }
  }, [config, isProcessing]);

  const handleConfigChange = async (newConfig: Partial<QueryConfig>) => {
    const updatedConfig = { ...config, ...newConfig };
    setConfig(updatedConfig);
    
    try {
      await apiClient.updateConfig(updatedConfig);
    } catch (error) {
      console.error('Failed to update config:', error);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Header */}
      <header className="flex-shrink-0 bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-600 rounded-xl flex items-center justify-center shadow-md">
                <span className="text-white text-xl font-bold">📘</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  강의계획서 RAG 챗봇
                </h1>
                <p className="text-sm text-gray-500">
                  AI 기반 강의계획서 검색 시스템
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Server Status */}
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-gray-100"
              >
                {serverStatus === 'checking' && (
                  <>
                    <Loader2 className="w-4 h-4 text-gray-400 animate-spin" />
                    <span className="text-sm text-gray-600">연결 확인 중...</span>
                  </>
                )}
                {serverStatus === 'healthy' && (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-green-700">연결됨</span>
                  </>
                )}
                {serverStatus === 'error' && (
                  <>
                    <AlertCircle className="w-4 h-4 text-red-500" />
                    <span className="text-sm text-red-700">연결 실패</span>
                  </>
                )}
              </motion.div>

              {/* Settings Button */}
              <button
                onClick={() => setIsSettingsOpen(!isSettingsOpen)}
                className="p-2 rounded-lg hover:bg-gray-100 transition-colors relative"
              >
                <Settings className="w-5 h-5 text-gray-600" />
                {isSettingsOpen && (
                  <span className="absolute top-1 right-1 w-2 h-2 bg-primary-500 rounded-full" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Settings Panel */}
        <AnimatePresence>
          {isSettingsOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="border-t border-gray-200 bg-gray-50 overflow-hidden"
            >
              <div className="max-w-7xl mx-auto px-4 py-4">
                <h3 className="text-sm font-semibold text-gray-900 mb-3">
                  ⚙️ 검색 설정
                </h3>
                <div className="space-y-3">
                  <label className="flex items-start gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.use_reranking}
                      onChange={(e) => handleConfigChange({ use_reranking: e.target.checked })}
                      className="mt-1 w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                    />
                    <div>
                      <div className="text-sm font-medium text-gray-900">
                        재랭킹 사용 (Reranking)
                      </div>
                      <div className="text-xs text-gray-500">
                        Cross-encoder를 사용하여 검색 결과를 더 정확하게 재정렬합니다. 
                        정확도는 높아지지만 속도가 약간 느려집니다.
                      </div>
                    </div>
                  </label>

                  <label className="flex items-start gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={config.use_query_expansion}
                      onChange={(e) => handleConfigChange({ use_query_expansion: e.target.checked })}
                      className="mt-1 w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                    />
                    <div>
                      <div className="text-sm font-medium text-gray-900">
                        질문 변환/확장 사용 (Query Transformation/Expansion)
                      </div>
                      <div className="text-xs text-gray-500">
                        질문을 자동으로 명확하게 변환하고 여러 관점에서 확장하여 검색합니다. 
                        오타나 모호한 표현도 잘 처리합니다.
                      </div>
                    </div>
                  </label>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* Chat Interface */}
      <main className="flex-1 overflow-hidden">
        {serverStatus === 'error' ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
              <h2 className="text-xl font-bold text-gray-900 mb-2">
                서버에 연결할 수 없습니다
              </h2>
              <p className="text-gray-600 mb-4">
                백엔드 서버가 실행 중인지 확인해주세요.
              </p>
              <button
                onClick={checkHealth}
                className="px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors"
              >
                다시 연결
              </button>
            </div>
          </div>
        ) : (
          <ChatInterface
            messages={messages}
            onSendMessage={handleSendMessage}
            isProcessing={isProcessing}
          />
        )}
      </main>
    </div>
  );
}

export default App;

