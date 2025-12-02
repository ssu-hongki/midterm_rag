export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  status?: 'sending' | 'processing' | 'complete' | 'error';
  progress?: ProgressStatus;
  contexts?: Context[];
  transformedQuery?: string;
  filters?: Record<string, string>;
  expandedQueries?: string[];
}

export interface ProgressStatus {
  step: 'analyzing' | 'filtering' | 'searching' | 'reranking' | 'generating' | 'complete';
  message: string;
  details?: string;
}

export interface Context {
  강좌명: string;
  과목코드: string;
  담당교수: string;
  source_pdf: string;
  chunk_id: string;
  score: number;
  text_preview?: string;
}

export interface QueryConfig {
  use_reranking: boolean;
  use_query_expansion: boolean;
}

export interface ChatHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface SSEEvent {
  type: 'status' | 'transformed_query' | 'filters' | 'expanded_queries' | 
        'contexts_found' | 'answer_start' | 'answer_chunk' | 'answer_complete' | 
        'contexts' | 'complete' | 'error';
  step?: string;
  message?: string;
  original?: string;
  transformed?: string;
  filters?: Record<string, string>;
  queries?: string[];
  count?: number;
  content?: string;
  full_answer?: string;
  contexts?: Context[];
}

