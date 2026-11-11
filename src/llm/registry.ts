// Picks adapters from CHAT_PROVIDER and EMBED_PROVIDER (docs/PLAN.md M2) and, given a sink, wraps them in call
// logging (M3).
import type { Config } from '../config';
import { type LlmCallSink, LoggedChatProvider, LoggedEmbeddingProvider } from '../obs/logger';
import { FAKE_CHAT_MODEL, FAKE_JUDGE_MODEL, FakeChatProvider, FakeEmbeddingProvider } from './fake';
import { OpenAICompatibleChatProvider, OpenAICompatibleEmbeddingProvider } from './openaiCompatible';
import { pricing, priceFor } from './pricing';
import type { ChatProvider, EmbeddingProvider } from './types';

function createInnerChatProvider(config: Config): {
  inner: ChatProvider;
  chatModel: string;
  judgeModel: string;
} {
  switch (config.chatProvider) {
    case 'ollama':
      return {
        inner: new OpenAICompatibleChatProvider({
          baseURL: config.baseURL,
          chatModel: config.chatModel,
          judgeModel: config.judgeModel,
        }),
        chatModel: config.chatModel,
        judgeModel: config.judgeModel,
      };
    case 'fake':
      return { inner: new FakeChatProvider(), chatModel: FAKE_CHAT_MODEL, judgeModel: FAKE_JUDGE_MODEL };
  }
}

function createInnerEmbeddingProvider(config: Config): EmbeddingProvider {
  switch (config.embedProvider) {
    case 'ollama':
      return new OpenAICompatibleEmbeddingProvider(config.embedModel, config.embedding, {
        baseURL: config.baseURL,
      });
    case 'fake':
      return new FakeEmbeddingProvider(config.embedModel, config.embedding.dim);
  }
}

/** With a sink, every call is logged; prices are resolved here so an unpriced model fails before any call. */
export function createChatProvider(config: Config, sink?: LlmCallSink): ChatProvider {
  const { inner, chatModel, judgeModel } = createInnerChatProvider(config);
  if (!sink) {
    return inner;
  }
  return new LoggedChatProvider(inner, {
    sink,
    provider: config.chatProvider,
    chatModel,
    judgeModel,
    chatPrice: priceFor(pricing, chatModel),
    judgePrice: priceFor(pricing, judgeModel),
  });
}

export function createEmbeddingProvider(config: Config, sink?: LlmCallSink): EmbeddingProvider {
  const inner = createInnerEmbeddingProvider(config);
  if (!sink) {
    return inner;
  }
  return new LoggedEmbeddingProvider(inner, {
    sink,
    provider: config.embedProvider,
    price: priceFor(pricing, inner.model),
  });
}
