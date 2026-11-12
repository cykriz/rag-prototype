import type { Sql } from 'postgres';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';

import { type Config, loadConfig } from '../../src/config';
import { FakeChatProvider } from '../../src/llm/fake';
import { createChatProvider, createEmbeddingProvider } from '../../src/llm/registry';
import { withRetry } from '../../src/llm/retry';
import { countTokens } from '../../src/llm/tokens';
import type { ChatProvider, EmbeddingProvider } from '../../src/llm/types';
import {
  type LlmCallRecord,
  type LlmCallSink,
  LoggedChatProvider,
  LoggedEmbeddingProvider,
  PostgresLlmCallSink,
  toLlmCallRow,
} from '../../src/obs/logger';

class MemorySink implements LlmCallSink {
  readonly records: LlmCallRecord[] = [];
  async write(record: LlmCallRecord): Promise<void> {
    this.records.push(record);
  }
}

const FREE = { inputPerMTok: 0, outputPerMTok: 0 };
const PAID = { inputPerMTok: 3, outputPerMTok: 15 };
const Answer = z.object({ answer: z.string() });
const REQ = { messages: [{ role: 'user' as const, content: 'hello world' }] };
const FAKE_CONFIG: Config = loadConfig({ CHAT_PROVIDER: 'fake', EMBED_PROVIDER: 'fake' });

function fakeChat(script: ConstructorParameters<typeof FakeChatProvider>[0] = []) {
  const sink = new MemorySink();
  const provider = new LoggedChatProvider(new FakeChatProvider(script), {
    sink,
    provider: 'fake',
    chatModel: 'fake-chat',
    judgeModel: 'fake-judge',
    chatPrice: FREE,
    judgePrice: FREE,
  });
  return { sink, provider };
}

/** Stub reporting usage, which the fakes never do. */
const usageChat: ChatProvider = {
  generate: async () => ({
    text: 'hi',
    toolCalls: [],
    model: 'paid-chat',
    attempts: 2,
    usage: { inputTokens: 1000, outputTokens: 500 },
  }),
  generateObject: () => Promise.reject(new Error('unused')),
  judge: () => Promise.reject(new Error('unused')),
};

function expectOneRecord(sink: MemorySink): LlmCallRecord {
  expect(sink.records).toHaveLength(1);
  const record = sink.records[0] as LlmCallRecord;
  expect(Number.isInteger(record.latencyMs)).toBe(true);
  expect(record.latencyMs).toBeGreaterThanOrEqual(0);
  return record;
}

describe('LoggedChatProvider', () => {
  it('logs provider usage, attempts and cost', async () => {
    const sink = new MemorySink();
    const provider = new LoggedChatProvider(usageChat, {
      sink,
      provider: 'stub',
      chatModel: 'paid-chat',
      judgeModel: 'paid-chat',
      chatPrice: PAID,
      judgePrice: PAID,
    });
    const res = await provider.generate(REQ);
    expect(res.text).toBe('hi');
    expect(expectOneRecord(sink)).toMatchObject({
      operation: 'generate',
      provider: 'stub',
      model: 'paid-chat',
      inputTokens: 1000,
      outputTokens: 500,
      tokenSource: 'usage',
      attempts: 2,
      status: 'ok',
    });
    expect(sink.records[0]?.costUsd).toBeCloseTo(0.0105, 12);
  });

  it('estimates tokens with tiktoken when the provider reports no usage', async () => {
    const { sink, provider } = fakeChat(['some reply']);
    await provider.generate(REQ);
    expect(expectOneRecord(sink)).toMatchObject({
      model: 'fake-chat',
      inputTokens: countTokens('hello world'),
      outputTokens: countTokens('some reply'),
      tokenSource: 'tiktoken_estimate',
      costUsd: 0,
      attempts: 1,
      status: 'ok',
    });
  });

  it('logs generateObject with summed attempts and the value as output', async () => {
    const { sink, provider } = fakeChat(['{bad', '{"answer":"x"}']);
    const res = await provider.generateObject(REQ, Answer);
    expect(res.status).toBe('ok');
    expect(expectOneRecord(sink)).toMatchObject({
      operation: 'generate',
      attempts: 2,
      status: 'ok',
      outputTokens: countTokens('{"answer":"x"}'),
    });
  });

  it('logs invalid_output after the repair rounds', async () => {
    const { sink, provider } = fakeChat(['{bad', '{bad', '{bad']);
    const res = await provider.generateObject(REQ, Answer);
    expect(res.status).toBe('invalid_output');
    expect(expectOneRecord(sink)).toMatchObject({ status: 'invalid_output', attempts: 3 });
  });

  it('logs judge calls as operation judge on the judge model', async () => {
    const { sink, provider } = fakeChat(['{"answer":"x"}']);
    await provider.judge(REQ, Answer);
    expect(expectOneRecord(sink)).toMatchObject({ operation: 'judge', model: 'fake-judge', status: 'ok' });
  });

  it('logs a thrown provider error once and rethrows it', async () => {
    const boom = new Error('boom');
    const { sink, provider } = fakeChat([boom]);
    await expect(provider.generate(REQ)).rejects.toBe(boom);
    expect(expectOneRecord(sink)).toMatchObject({
      operation: 'generate',
      model: 'fake-chat',
      inputTokens: countTokens('hello world'),
      outputTokens: null,
      tokenSource: 'tiktoken_estimate',
      costUsd: 0,
      attempts: 1,
      status: 'error',
    });
  });

  it('logs the attempts of a call that exhausted its retries', async () => {
    const sink = new MemorySink();
    const serverError = () => Promise.reject(Object.assign(new Error('unavailable'), { status: 503 }));
    const retrying: ChatProvider = {
      ...usageChat,
      generate: async () => (await withRetry(serverError, { sleep: async () => {} })).value,
    };
    const provider = new LoggedChatProvider(retrying, {
      sink,
      provider: 'stub',
      chatModel: 'paid-chat',
      judgeModel: 'paid-chat',
      chatPrice: PAID,
      judgePrice: PAID,
    });
    await expect(provider.generate(REQ)).rejects.toThrow('unavailable');
    expect(expectOneRecord(sink)).toMatchObject({ status: 'error', attempts: 4, costUsd: 0 });
  });

  it('keeps the provider error when the sink fails on the error path', async () => {
    const boom = new Error('boom');
    const sinkErr = new Error('sink down');
    const provider = new LoggedChatProvider(new FakeChatProvider([boom]), {
      sink: { write: () => Promise.reject(sinkErr) },
      provider: 'fake',
      chatModel: 'fake-chat',
      judgeModel: 'fake-judge',
      chatPrice: FREE,
      judgePrice: FREE,
    });
    const err = await provider.generate(REQ).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(AggregateError);
    expect((err as AggregateError).errors).toEqual([boom, sinkErr]);
  });
});

describe('LoggedEmbeddingProvider', () => {
  it('logs one embed record with estimated input and zero output tokens', async () => {
    const sink = new MemorySink();
    const inner: EmbeddingProvider = {
      model: 'emb',
      dim: 2,
      embed: async (texts) => ({ vectors: texts.map(() => [1, 0]), model: 'emb', attempts: 1 }),
    };
    const provider = new LoggedEmbeddingProvider(inner, { sink, provider: 'stub', price: FREE });
    expect(provider).toMatchObject({ model: 'emb', dim: 2 });
    const res = await provider.embed(['hello world', 'bun'], 'document');
    expect(res.vectors).toHaveLength(2);
    expect(expectOneRecord(sink)).toMatchObject({
      operation: 'embed',
      model: 'emb',
      inputTokens: countTokens('hello world') + countTokens('bun'),
      outputTokens: 0,
      tokenSource: 'tiktoken_estimate',
      status: 'ok',
    });
  });
});

describe('registry with a sink', () => {
  it('wraps the fakes so every call is logged', async () => {
    const sink = new MemorySink();
    await createChatProvider(FAKE_CONFIG, sink).generate(REQ);
    await createEmbeddingProvider(FAKE_CONFIG, sink).embed(['a'], 'query');
    expect(sink.records.map((r) => [r.operation, r.provider, r.model])).toEqual([
      ['generate', 'fake', 'fake-chat'],
      ['embed', 'fake', 'nomic-embed-text'],
    ]);
  });

  it('returns the bare provider without a sink', () => {
    expect(createChatProvider(FAKE_CONFIG)).toBeInstanceOf(FakeChatProvider);
  });

  it('fails before any call when a configured model has no price', () => {
    const config: Config = { ...loadConfig({}), chatModel: 'unpriced-model' };
    expect(() => createChatProvider(config, new MemorySink())).toThrow('No price for model "unpriced-model"');
  });
});

describe('Postgres sink mapping', () => {
  const record: LlmCallRecord = {
    operation: 'generate',
    provider: 'ollama',
    model: 'qwen2.5:7b',
    inputTokens: 12,
    outputTokens: 34,
    tokenSource: 'usage',
    latencyMs: 56,
    costUsd: 0.0007,
    attempts: 2,
    status: 'ok',
  };

  it('maps every record field to its llm_calls column and adds run and request id', () => {
    expect(
      toLlmCallRow(record, { runId: 'run-1', requestId: '00000000-0000-4000-8000-000000000001' }),
    ).toEqual({
      run_id: 'run-1',
      request_id: '00000000-0000-4000-8000-000000000001',
      operation: 'generate',
      provider: 'ollama',
      model: 'qwen2.5:7b',
      input_tokens: 12,
      output_tokens: 34,
      token_source: 'usage',
      latency_ms: 56,
      cost_usd: 0.0007,
      attempts: 2,
      status: 'ok',
    });
  });

  it('keeps an error record as is and defaults request_id to null', () => {
    const error: LlmCallRecord = {
      ...record,
      outputTokens: null,
      tokenSource: 'tiktoken_estimate',
      costUsd: 0,
      status: 'error',
    };
    expect(toLlmCallRow(error, { runId: 'run-1' })).toMatchObject({
      request_id: null,
      input_tokens: 12,
      output_tokens: null,
      token_source: 'tiktoken_estimate',
      cost_usd: 0,
      status: 'error',
    });
  });

  it('rejects an empty runId', () => {
    // The constructor validates before touching the connection.
    expect(() => new PostgresLlmCallSink({} as Sql, { runId: ' ' })).toThrow('non-empty runId');
  });
});
