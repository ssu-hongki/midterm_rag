import { QueryConfig, SSEEvent, ChatHistoryMessage } from './types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export class RAGAPIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      const data = await response.json();
      return data.status === 'healthy' && data.rag_chain_ready;
    } catch (error) {
      console.error('Health check failed:', error);
      return false;
    }
  }

  async updateConfig(config: QueryConfig): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error('Failed to update config');
    }
  }

  async *queryStream(
    query: string, 
    config: QueryConfig,
    chatHistory: ChatHistoryMessage[] = []
  ): AsyncGenerator<SSEEvent, void, unknown> {
    const response = await fetch(`${this.baseURL}/api/query/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        ...config,
        chat_history: chatHistory,
      }),
    });

    if (!response.ok) {
      throw new Error(`Query failed: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // 마지막 줄은 불완전할 수 있으므로 보관
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const event: SSEEvent = JSON.parse(data);
              yield event;
            } catch (error) {
              console.error('Failed to parse SSE event:', error);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

export const apiClient = new RAGAPIClient();

