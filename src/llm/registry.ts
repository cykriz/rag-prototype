// Picks adapters from CHAT_PROVIDER and EMBED_PROVIDER (docs/PLAN.md M2).
import type { Config } from '../config';
import { FakeChatProvider, FakeEmbeddingProvider } from './fake';
import { OpenAICompatibleChatProvider, OpenAICompatibleEmbeddingProvider } from './openaiCompatible';
import type { ChatProvider, EmbeddingProvider } from './types';

export function createChatProvider(config: Config): ChatProvider {
  switch (config.chatProvider) {
    case 'ollama':
      return new OpenAICompatibleChatProvider({
        baseURL: config.baseURL,
        chatModel: config.chatModel,
        judgeModel: config.judgeModel,
      });
    case 'fake':
      return new FakeChatProvider();
  }
}

export function createEmbeddingProvider(config: Config): EmbeddingProvider {
  switch (config.embedProvider) {
    case 'ollama':
      return new OpenAICompatibleEmbeddingProvider(config.embedModel, config.embedding, {
        baseURL: config.baseURL,
      });
    case 'fake':
      return new FakeEmbeddingProvider(config.embedModel, config.embedding.dim);
  }
}
