import { describe, expect, it } from 'vitest';
import type { z } from 'zod';

import { formatExpansion } from '../../src/cli/retrieve';
import { FakeChatProvider } from '../../src/llm/fake';
import type { ChatProvider, GenerateRequest } from '../../src/llm/types';
import { type LlmCallRecord, type LlmCallSink, LoggedChatProvider } from '../../src/obs/logger';
import { dedupeExpansions, expandQuery, expandRecord, searchQueries } from '../../src/rag/expand';
import { MAX_EXPANSIONS } from '../../src/rag/pipeline';
import { EXPAND_SEED, EXPAND_SYSTEM_PROMPT, EXPAND_TEMPERATURE } from '../../src/rag/prompts';
import { MAX_QUERY_CHARS } from '../../src/rag/schemas';
import { ChatStageError } from '../../src/rag/stage';
import { fuseHits } from '../../src/retrieval/multi';
import { RRF_K } from '../../src/retrieval/rrf';
import type { SearchHit } from '../../src/retrieval/search';

const QUESTION = 'how do I add a package and run tests?';
const ok = (queries: unknown) => JSON.stringify({ queries });

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

const FALLBACK = { queries: [], dropped: 0, fallback: true };

describe('expandQuery', () => {
  it('returns the expansions and logs one call as operation expand', async () => {
    const { sink, fake, chat } = logged([ok(['bun add dependency', 'bun test runner'])]);
    const trace = await expandQuery(chat, QUESTION);
    expect(trace).toEqual({
      queries: ['bun add dependency', 'bun test runner'],
      dropped: 0,
      fallback: false,
      repairs: 0,
    });
    expect(sink.records.map((r) => [r.operation, r.model, r.status])).toEqual([
      ['expand', 'fake-chat', 'ok'],
    ]);
    expect(fake.calls[0]).toEqual([
      { role: 'system', content: EXPAND_SYSTEM_PROMPT },
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
          value: schema.parse({ queries: ['q'] }),
        };
      },
      judge: () => Promise.reject(new Error('unused')),
    };
    await expandQuery(spy, QUESTION);
    expect(seen[0]).toMatchObject({
      operation: 'expand',
      temperature: EXPAND_TEMPERATURE,
      seed: EXPAND_SEED,
    });
    expect([EXPAND_TEMPERATURE, EXPAND_SEED, MAX_EXPANSIONS]).toEqual([0, 42, 3]);
  });

  it('trims the expansions', async () => {
    const { chat } = logged([ok(['  bun add  '])]);
    expect((await expandQuery(chat, QUESTION)).queries).toEqual(['bun add']);
  });

  it('repairs an invalid reply and uses the repaired output', async () => {
    const { sink, chat } = logged(['{not json', ok(['bun add'])]);
    expect(await expandQuery(chat, QUESTION)).toEqual({
      queries: ['bun add'],
      dropped: 0,
      fallback: false,
      repairs: 1,
    });
    expect(sink.records[0]).toMatchObject({ operation: 'expand', status: 'ok' });
  });

  it('falls back to the question alone on invalid JSON after the repairs', async () => {
    const { sink, chat } = logged(['{bad', '{bad', '{bad']);
    expect(await expandQuery(chat, QUESTION)).toEqual({ ...FALLBACK, repairs: 2 });
    expect(sink.records[0]).toMatchObject({ operation: 'expand', status: 'invalid_output' });
  });

  it.each([
    ['a schema miss', JSON.stringify({ expansions: ['bun add'] })],
    ['an empty list', ok([])],
    [`more than ${MAX_EXPANSIONS} queries`, ok(['a', 'b', 'c', 'd'])],
    ['a blank query', ok(['bun add', '   '])],
    ['a query over the length cap', ok(['x'.repeat(MAX_QUERY_CHARS + 1)])],
    ['a non-string query', ok([42])],
  ])('falls back after the repairs on %s', async (_, reply) => {
    const { chat } = logged([reply, reply, reply]);
    expect(await expandQuery(chat, QUESTION)).toEqual({ ...FALLBACK, repairs: 2 });
  });

  it('drops duplicates of the question and of earlier expansions, case and whitespace aside', async () => {
    const { chat } = logged([ok(['bun add', ` HOW do I add a package   and run tests? `, 'Bun  ADD'])]);
    expect(await expandQuery(chat, QUESTION)).toEqual({
      queries: ['bun add'],
      dropped: 2,
      fallback: false,
      repairs: 0,
    });
  });

  it('keeps no query but does not fall back when every expansion repeats the question', async () => {
    const { chat } = logged([ok([QUESTION])]);
    const trace = await expandQuery(chat, QUESTION);
    expect(trace).toEqual({ queries: [], dropped: 1, fallback: false, repairs: 0 });
    expect(searchQueries(QUESTION, trace)).toEqual([QUESTION]);
    expect(formatExpansion(trace)).toBe(
      'expand: no new queries (1 duplicate dropped), searching with the question only',
    );
  });

  it('wraps a provider error in ChatStageError', async () => {
    const boom = new Error('chat down');
    const err = await expandQuery(new FakeChatProvider([boom]), QUESTION).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ChatStageError);
    expect(err).toMatchObject({ stage: 'expand' });
    expect((err as ChatStageError).cause).toBe(boom);
  });
});

describe('expand helpers', () => {
  it('searchQueries puts the question first', () => {
    expect(searchQueries('q', { queries: ['a', 'b'] })).toEqual(['q', 'a', 'b']);
    expect(searchQueries('q', null)).toEqual(['q']);
  });

  it('dedupeExpansions keeps the first spelling', () => {
    expect(dedupeExpansions('Q', ['a', 'A ', 'q', 'b'])).toEqual(['a', 'b']);
  });

  it('expandRecord drops the repairs', () => {
    expect(expandRecord({ queries: ['a'], dropped: 0, fallback: false, repairs: 1 })).toEqual({
      queries: ['a'],
      dropped: 0,
      fallback: false,
    });
    expect(expandRecord(null)).toBeNull();
  });

  it('formatExpansion lists the queries or names the fallback', () => {
    expect(formatExpansion({ queries: ['a', 'b'], dropped: 0, fallback: false, repairs: 0 })).toBe(
      'expand: 2 queries besides the question\n  a\n  b',
    );
    expect(formatExpansion({ ...FALLBACK, repairs: 2 })).toBe(
      'expand: invalid output after 2 repairs, searching with the question only',
    );
  });
});

const hit = (chunkId: number, score = 0.5): SearchHit => ({
  chunkId,
  sourcePath: `doc-${chunkId}.md`,
  headingPath: '',
  content: `content ${chunkId}`,
  score,
});

describe('fuseHits', () => {
  it('ranks a chunk found by several queries above one found once', () => {
    // 3 is only third in each list but appears in all three; 1 is first, but in one list only.
    const fused = fuseHits(
      [
        [hit(1), hit(2), hit(3)],
        [hit(4), hit(5), hit(3)],
        [hit(6), hit(7), hit(3)],
      ],
      3,
    );
    expect(fused.map((h) => h.chunkId)).toEqual([3, 1, 4]);
    expect(fused[0]?.score).toBeCloseTo(3 / (RRF_K + 3));
  });

  it('breaks ties by chunk id, cuts to k and keeps content of the first occurrence', () => {
    const fused = fuseHits([[hit(9, 0.9)], [hit(2, 0.1)]], 1);
    expect(fused).toEqual([{ ...hit(2), score: 1 / (RRF_K + 1) }]);
  });

  it('keeps the order of a single list', () => {
    expect(fuseHits([[hit(5), hit(1), hit(3)]], 5).map((h) => h.chunkId)).toEqual([5, 1, 3]);
  });
});
