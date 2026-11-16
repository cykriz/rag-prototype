import { describe, expect, it } from 'vitest';
import type { z } from 'zod';

import { formatRerank } from '../../src/cli/retrieve';
import { FakeChatProvider } from '../../src/llm/fake';
import type { ChatProvider, GenerateRequest } from '../../src/llm/types';
import { type LlmCallRecord, type LlmCallSink, LoggedChatProvider } from '../../src/obs/logger';
import {
  MAX_RERANK_SCORE,
  RERANK_CANDIDATES,
  RERANK_CONTEXT_CHARS,
  retrievalDepth,
} from '../../src/rag/pipeline';
import {
  buildRerankMessages,
  RERANK_SEED,
  RERANK_SYSTEM_PROMPT,
  RERANK_TEMPERATURE,
  type RerankCandidate,
  rerankChunkChars,
} from '../../src/rag/prompts';
import { applyRanking, type Reranker, rerankChunks, rerankItems, rerankRecord } from '../../src/rag/rerank';
import { ChatStageError } from '../../src/rag/stage';

const QUESTION = 'how do I read an environment variable?';

const candidate = (chunkId: number, content = `content of chunk ${chunkId}`): RerankCandidate => ({
  chunkId,
  sourcePath: 'runtime/env.md',
  headingPath: 'Reading',
  content,
});
/** Candidates 1, 2, 3, 4 in search order. */
const CANDIDATES = [1, 2, 3, 4].map((id) => candidate(id));
const IDS = CANDIDATES.map((c) => c.chunkId);

const ranking = (...entries: [number, number][]) =>
  JSON.stringify({ ranking: entries.map(([chunk_id, score]) => ({ chunk_id, score })) });

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

const FALLBACK = {
  candidates: IDS,
  order: IDS,
  scores: [null, null, null, null],
  unknown: 0,
  duplicates: 0,
  missing: 0,
  fallback: true,
  repairs: 2,
};

describe('rerankChunks', () => {
  it('orders by score, keeps the model order on ties and logs one call as operation rerank', async () => {
    const { sink, fake, chat } = logged([ranking([2, 3], [4, 9], [1, 3], [3, 7])]);
    const trace = await rerankChunks(chat, QUESTION, CANDIDATES);
    expect(trace).toEqual({
      candidates: IDS,
      order: [4, 3, 2, 1],
      scores: [9, 7, 3, 3],
      unknown: 0,
      duplicates: 0,
      missing: 0,
      fallback: false,
      repairs: 0,
    });
    expect(sink.records.map((r) => [r.operation, r.model, r.status])).toEqual([
      ['rerank', 'fake-chat', 'ok'],
    ]);
    expect(fake.calls[0]).toEqual(buildRerankMessages(QUESTION, CANDIDATES));
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
          value: schema.parse({ ranking: [{ chunk_id: 1, score: 5 }] }),
        };
      },
      judge: () => Promise.reject(new Error('unused')),
    };
    await rerankChunks(spy, QUESTION, CANDIDATES);
    expect(seen[0]).toMatchObject({
      operation: 'rerank',
      temperature: RERANK_TEMPERATURE,
      seed: RERANK_SEED,
    });
    expect([RERANK_TEMPERATURE, RERANK_SEED, MAX_RERANK_SCORE, RERANK_CANDIDATES]).toEqual([0, 42, 10, 20]);
  });

  it('repairs an invalid reply and uses the repaired output', async () => {
    const { sink, chat } = logged(['{not json', ranking([3, 8])]);
    const trace = await rerankChunks(chat, QUESTION, CANDIDATES);
    expect(trace).toMatchObject({ order: [3, 1, 2, 4], fallback: false, repairs: 1 });
    expect(sink.records[0]).toMatchObject({ operation: 'rerank', status: 'ok' });
  });

  it('falls back to the search order on invalid JSON after the repairs', async () => {
    const { sink, chat } = logged(['{bad', '{bad', '{bad']);
    expect(await rerankChunks(chat, QUESTION, CANDIDATES)).toEqual(FALLBACK);
    expect(sink.records[0]).toMatchObject({ operation: 'rerank', status: 'invalid_output' });
  });

  it.each([
    ['a schema miss', JSON.stringify({ order: [1, 2] })],
    ['an empty ranking', ranking()],
    ['a non-integer ID', ranking([1.5, 4])],
    ['a score above the range', ranking([1, MAX_RERANK_SCORE + 1])],
    ['a negative score', ranking([1, -1])],
    ['a ranking that names no candidate', ranking([98, 5], [99, 4])],
  ])('falls back after the repairs on %s', async (_, reply) => {
    const { chat } = logged([reply, reply, reply]);
    expect(await rerankChunks(chat, QUESTION, CANDIDATES)).toEqual(FALLBACK);
  });

  it('tells the model the allowed IDs when a ranking names no candidate', async () => {
    const { fake, chat } = logged([ranking([98, 5]), ranking([2, 5])]);
    await rerankChunks(chat, QUESTION, CANDIDATES);
    expect(fake.calls[1]?.at(-1)?.content).toContain('allowed: 1, 2, 3, 4');
  });

  it('ignores unknown IDs, keeps the first of duplicates and appends missing ones in search order', async () => {
    const { chat } = logged([ranking([3, 6], [77, 9], [3, 10], [1, 2])]);
    expect(await rerankChunks(chat, QUESTION, CANDIDATES)).toEqual({
      candidates: IDS,
      order: [3, 1, 2, 4],
      scores: [6, 2, null, null],
      unknown: 1,
      duplicates: 1,
      missing: 2,
      fallback: false,
      repairs: 0,
    });
  });

  it.each([
    ['no candidates', []],
    ['one candidate', [candidate(7)]],
  ])('makes no call with %s', async (_, candidates) => {
    const { sink, chat } = logged([]);
    const trace = await rerankChunks(chat, QUESTION, candidates);
    const ids = candidates.map((c) => c.chunkId);
    expect(trace).toEqual({
      candidates: ids,
      order: ids,
      scores: ids.map(() => null),
      unknown: 0,
      duplicates: 0,
      missing: 0,
      fallback: false,
      repairs: 0,
    });
    expect(sink.records).toEqual([]);
  });

  it('wraps a provider error as ChatStageError of stage rerank', async () => {
    const { chat } = logged([new Error('chat down')]);
    const err = await rerankChunks(chat, QUESTION, CANDIDATES).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ChatStageError);
    expect(err).toMatchObject({ stage: 'rerank', message: 'chat down' });
  });
});

describe('applyRanking', () => {
  it('appends every candidate when the ranking is all unknown', () => {
    expect(applyRanking([1, 2], [{ chunk_id: 9, score: 1 }])).toEqual({
      order: [1, 2],
      scores: [null, null],
      unknown: 1,
      duplicates: 0,
      missing: 2,
    });
  });
});

describe('rerankItems', () => {
  const reranker =
    (order: number[]): Reranker =>
    async (_, candidates) => ({
      candidates: candidates.map((c) => c.chunkId),
      order,
      scores: order.map(() => 1),
      unknown: 0,
      duplicates: 0,
      missing: 0,
      fallback: false,
      repairs: 0,
    });

  it('keeps the top k items in rerank order', async () => {
    const { kept, trace } = await rerankItems(reranker([4, 2, 1, 3]), QUESTION, CANDIDATES, (c) => c, 2);
    expect(kept.map((c) => c.chunkId)).toEqual([4, 2]);
    expect(rerankRecord(trace)).not.toHaveProperty('repairs');
  });

  it('keeps fewer than k when there are fewer candidates', async () => {
    const { kept } = await rerankItems(reranker([2, 1]), QUESTION, CANDIDATES.slice(0, 2), (c) => c, 5);
    expect(kept.map((c) => c.chunkId)).toEqual([2, 1]);
  });
});

describe('rerank prompt', () => {
  it('cuts each chunk to its share of the budget and ends the cut with an ellipsis', () => {
    const long = Array.from({ length: RERANK_CANDIDATES }, (_, i) =>
      candidate(i + 1, `word${i} `.repeat(1000)),
    );
    const max = rerankChunkChars(RERANK_CANDIDATES);
    expect(max).toBe(RERANK_CONTEXT_CHARS / RERANK_CANDIDATES);
    const user = buildRerankMessages(QUESTION, long)[1]?.content ?? '';
    const bodies = user
      .split('\n\n')
      .filter((block) => block.startsWith('[chunk:'))
      .map((block) => block.split('\n')[1] ?? '');
    expect(bodies).toHaveLength(RERANK_CANDIDATES);
    for (const body of bodies) {
      expect(body.length).toBeLessThanOrEqual(max);
      expect(body.endsWith('…')).toBe(true);
    }
    expect(bodies.reduce((n, b) => n + b.length, 0)).toBeLessThanOrEqual(RERANK_CONTEXT_CHARS);
  });

  it('shows header, source label and collapsed content, then the question', () => {
    const [system, user] = buildRerankMessages(QUESTION, [candidate(5, 'Bun.env\n\n  reads   .env')]);
    expect(system?.content).toBe(RERANK_SYSTEM_PROMPT);
    expect(RERANK_SYSTEM_PROMPT).toContain(`0 to ${MAX_RERANK_SCORE}`);
    expect(user?.content).toBe(
      `Excerpts:\n\n[chunk:5] (runtime/env.md > Reading)\nBun.env reads .env\n\nQuestion: ${QUESTION}`,
    );
  });
});

describe('retrievalDepth', () => {
  it('fetches max(k, 20) candidates with rerank on and k otherwise', () => {
    expect(retrievalDepth({ rerank: 'off' }, 5)).toBe(5);
    expect(retrievalDepth({ rerank: 'on' }, 5)).toBe(RERANK_CANDIDATES);
    expect(retrievalDepth({ rerank: 'on' }, 30)).toBe(30);
  });
});

describe('formatRerank', () => {
  const base = { candidates: IDS, unknown: 0, duplicates: 0, missing: 0, fallback: false, repairs: 0 };

  it('lists the kept chunks with score and search rank, and what was ignored', () => {
    const trace = { ...base, order: [3, 1, 2, 4], scores: [6, 2, null, null], unknown: 1, missing: 2 };
    expect(formatRerank(trace, 3)).toBe(
      [
        'rerank: 4 candidates, top 3 kept (IDs: 1 unknown, 2 missing)',
        '  #3  score 6  search rank 3',
        '  #1  score 2  search rank 1',
        '  #2  score -  search rank 2',
      ].join('\n'),
    );
  });

  it('names a fallback and a skip', () => {
    expect(formatRerank({ ...FALLBACK }, 2)).toBe(
      'rerank: invalid output after 2 repairs, keeping the search order',
    );
    expect(formatRerank({ ...base, candidates: [7], order: [7], scores: [null] }, 5)).toBe(
      'rerank: 1 candidate, nothing to rerank',
    );
  });
});
