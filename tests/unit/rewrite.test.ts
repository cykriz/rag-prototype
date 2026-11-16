import { describe, expect, it } from 'vitest';
import type { z } from 'zod';

import { FakeChatProvider } from '../../src/llm/fake';
import type { ChatProvider, GenerateRequest } from '../../src/llm/types';
import { type LlmCallRecord, type LlmCallSink, LoggedChatProvider } from '../../src/obs/logger';
import { REWRITE_SEED, REWRITE_SYSTEM_PROMPT, REWRITE_TEMPERATURE } from '../../src/rag/prompts';
import { rewriteQuery } from '../../src/rag/rewrite';
import { MAX_QUERY_CHARS } from '../../src/rag/schemas';
import { ChatStageError } from '../../src/rag/stage';

const QUESTION = 'how do I add a package?';
const ok = (rewritten: string) => JSON.stringify({ rewritten });

class MemorySink implements LlmCallSink {
  readonly records: LlmCallRecord[] = [];
  async write(record: LlmCallRecord): Promise<void> {
    this.records.push(record);
  }
}

const FREE = { inputPerMTok: 0, outputPerMTok: 0 };

function logged(script: ConstructorParameters<typeof FakeChatProvider>[0]) {
  const sink = new MemorySink();
  const fake = new FakeChatProvider(script);
  const chat = new LoggedChatProvider(fake, {
    sink,
    provider: 'fake',
    chatModel: 'fake-chat',
    judgeModel: 'fake-judge',
    chatPrice: FREE,
    judgePrice: FREE,
  });
  return { sink, fake, chat };
}

describe('rewriteQuery', () => {
  it('returns the rewrite and logs one call as operation rewrite', async () => {
    const { sink, fake, chat } = logged([ok('bun add package install dependency')]);
    const trace = await rewriteQuery(chat, QUESTION);
    expect(trace).toEqual({
      query: 'bun add package install dependency',
      fallback: false,
      repairs: 0,
    });
    expect(sink.records.map((r) => [r.operation, r.model, r.status])).toEqual([
      ['rewrite', 'fake-chat', 'ok'],
    ]);
    expect(fake.calls[0]).toEqual([
      { role: 'system', content: REWRITE_SYSTEM_PROMPT },
      { role: 'user', content: `Question: ${QUESTION}` },
    ]);
  });

  it('sends the fixed temperature and seed', async () => {
    const seen: GenerateRequest[] = [];
    const spy: ChatProvider = {
      generate: () => Promise.reject(new Error('unused')),
      generateObject: async <T>(req: GenerateRequest, schema: z.ZodType<T>) => {
        seen.push(req);
        return {
          model: 'spy',
          attempts: 1,
          repairs: 0,
          status: 'ok' as const,
          value: schema.parse({ rewritten: 'q' }),
        };
      },
      judge: () => Promise.reject(new Error('unused')),
    };
    await rewriteQuery(spy, QUESTION);
    expect(seen[0]).toMatchObject({
      operation: 'rewrite',
      temperature: REWRITE_TEMPERATURE,
      seed: REWRITE_SEED,
    });
    expect([REWRITE_TEMPERATURE, REWRITE_SEED]).toEqual([0, 42]);
  });

  it('trims the rewrite', async () => {
    const { chat } = logged([ok('  bun add  ')]);
    expect((await rewriteQuery(chat, QUESTION)).query).toBe('bun add');
  });

  it('repairs an invalid output without falling back', async () => {
    const { sink, chat } = logged(['{bad', ok('bun add')]);
    expect(await rewriteQuery(chat, QUESTION)).toMatchObject({
      query: 'bun add',
      fallback: false,
      repairs: 1,
    });
    expect(sink.records).toHaveLength(1);
  });

  it('falls back to the question on invalid JSON after the repairs', async () => {
    const { sink, chat } = logged(['{bad', '{bad', '{bad']);
    expect(await rewriteQuery(chat, QUESTION)).toEqual({
      query: QUESTION,
      fallback: true,
      repairs: 2,
    });
    expect(sink.records.map((r) => [r.operation, r.status])).toEqual([['rewrite', 'invalid_output']]);
  });

  it('falls back on a schema miss after the repairs', async () => {
    const miss = JSON.stringify({ query: 'bun add' });
    const { chat } = logged([miss, miss, miss]);
    expect(await rewriteQuery(chat, QUESTION)).toMatchObject({ query: QUESTION, fallback: true });
  });

  it.each([
    ['empty', ''],
    ['whitespace-only', ' \n\t '],
    ['over the length cap', 'x'.repeat(MAX_QUERY_CHARS + 1)],
  ])('falls back on %s text', async (_, text) => {
    const { chat } = logged([ok(text), ok(text), ok(text)]);
    expect(await rewriteQuery(chat, QUESTION)).toMatchObject({ query: QUESTION, fallback: true });
  });

  it('wraps a provider error instead of falling back', async () => {
    const boom = new Error('connection refused');
    const { sink, chat } = logged([boom]);
    const err = await rewriteQuery(chat, QUESTION).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ChatStageError);
    expect(err).toMatchObject({ stage: 'rewrite' });
    expect((err as ChatStageError).cause).toBe(boom);
    expect(sink.records.map((r) => [r.operation, r.status])).toEqual([['rewrite', 'error']]);
  });
});
