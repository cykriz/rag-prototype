import { Document } from '@langchain/core/documents';
import { APIConnectionError, APIError } from 'openai';
import postgres from 'postgres';
import { describe, expect, it } from 'vitest';

import { describeAnswerError, formatAnswer } from '../../src/cli/ask';
import { loadConfig } from '../../src/config';
import { FakeChatProvider } from '../../src/llm/fake';
import type { RagRequestRecord, RagRequestSink } from '../../src/obs/logger';
import { type AnswerDeps, answer, buildAnswerDeps, RetrievalError } from '../../src/rag/answer';
import { expanderFor } from '../../src/rag/expand';
import { RERANK_CANDIDATES } from '../../src/rag/pipeline';
import { buildAnswerMessages, buildRerankMessages, formatContext, NOT_FOUND } from '../../src/rag/prompts';
import { documentCandidate, rerankerFor } from '../../src/rag/rerank';
import { rewriterFor } from '../../src/rag/rewrite';
import { ChatStageError } from '../../src/rag/stage';
import { type ChunkMetadata, PgRetriever } from '../../src/retrieval/pgRetriever';

const doc = (chunk_id: number, source_path: string, heading_path: string, pageContent: string) =>
  new Document<ChunkMetadata>({
    pageContent,
    metadata: { chunk_id, source_path, heading_path, score: 1 - 0.1 * chunk_id },
  });

const DOCS = [
  doc(
    11,
    'runtime/env.md',
    'Environment variables > Reading',
    'Read variables with `process.env` or `Bun.env`.',
  ),
  doc(12, 'runtime/env.md', '', 'Bun reads `.env` files automatically.'),
];

const reply = (found: boolean, text: string, citations: number[]) =>
  JSON.stringify({ found, answer: text, citations });

class MemorySink implements RagRequestSink {
  readonly records: RagRequestRecord[] = [];
  async write(record: RagRequestRecord): Promise<void> {
    this.records.push(record);
  }
}

function setup(script: string[], docs = DOCS) {
  const chat = new FakeChatProvider(script);
  const requestSink = new MemorySink();
  const deps: AnswerDeps = {
    retriever: { invoke: async () => docs, invokeMany: async () => docs },
    chat,
    requestId: '00000000-0000-4000-8000-000000000001',
    runId: 'test-run',
    pipeline: 'baseline',
    requestSink,
  };
  return { chat, requestSink, deps };
}

describe('buildAnswerMessages', () => {
  it('renders the system rules and the context format', () => {
    expect(buildAnswerMessages('How do I read env vars?', formatContext(DOCS))).toMatchSnapshot();
  });
});

describe('answer', () => {
  it('returns a cited answer with source metadata', async () => {
    const { deps, requestSink } = setup([reply(true, 'Use Bun.env.', [11, 11])]);
    const res = await answer('How do I read env vars?', deps);
    expect(res).toMatchObject({
      found: true,
      answer: 'Use Bun.env.',
      citations: [
        { chunkId: 11, sourcePath: 'runtime/env.md', headingPath: 'Environment variables > Reading' },
      ],
      droppedCitations: [],
      retrievedIds: [11, 12],
      status: 'ok',
      repairs: 0,
    });
    expect(requestSink.records).toEqual([
      {
        requestId: deps.requestId,
        runId: 'test-run',
        questionId: null,
        pipeline: 'baseline',
        totalLatencyMs: res.latencyMs,
        retrievedIds: [11, 12],
        found: true,
      },
    ]);
    expect(res.latencyMs).toBeGreaterThanOrEqual(0);
    // The judge (M12) gets exactly the context the model saw.
    expect(res.context).toBe(formatContext(DOCS));
  });

  it('writes the golden question ID into the request row', async () => {
    const { deps, requestSink } = setup([reply(true, 'Use Bun.env.', [11])]);
    await answer('q', { ...deps, questionId: 'q007' });
    expect(requestSink.records[0]?.questionId).toBe('q007');
  });

  it('drops citations outside the context and reports invalid_output after the repairs', async () => {
    const invented = reply(true, 'Use Bun.env.', [11, 999]);
    const { deps, chat, requestSink } = setup([invented, invented, invented]);
    const res = await answer('How do I read env vars?', deps);
    expect(res).toMatchObject({
      found: true,
      citations: [{ chunkId: 11 }],
      droppedCitations: [999],
      status: 'invalid_output',
      repairs: 2,
    });
    // The repair message tells the model which ID is wrong.
    const repairMsg = chat.calls[1]?.at(-1)?.content;
    expect(repairMsg).toContain('citations 999 are not chunk IDs from the context (allowed: 11, 12)');
    expect(requestSink.records[0]?.found).toBe(true);
  });

  it('accepts an answer fixed in a repair round', async () => {
    const { deps } = setup([reply(true, 'Use Bun.env.', [999]), reply(true, 'Use Bun.env.', [12])]);
    const res = await answer('q', deps);
    expect(res).toMatchObject({
      status: 'ok',
      repairs: 1,
      citations: [{ chunkId: 12 }],
      droppedCitations: [],
    });
  });

  it('keeps found=true with zero valid citations, flagged as invalid_output', async () => {
    const invented = reply(true, 'Use Bun.env.', [998, 999]);
    const { deps } = setup([invented, invented, invented]);
    const res = await answer('q', deps);
    expect(res).toMatchObject({
      found: true,
      citations: [],
      droppedCitations: [998, 999],
      status: 'invalid_output',
    });
  });

  it('normalizes found=false to NOT_FOUND without citations', async () => {
    const { deps, chat, requestSink } = setup([reply(false, 'Bun has no such feature.', [11])]);
    const res = await answer('How do I configure the built-in GraphQL server?', deps);
    expect(res).toMatchObject({
      found: false,
      answer: NOT_FOUND,
      citations: [],
      droppedCitations: [],
      status: 'ok',
    });
    expect(chat.calls).toHaveLength(1);
    expect(requestSink.records[0]?.found).toBe(false);
  });

  it('treats found=true with answer NOT_FOUND as an abstention without a repair round', async () => {
    const { deps, chat } = setup([reply(true, ' NOT_FOUND ', [])]);
    const res = await answer('q', deps);
    expect(res).toMatchObject({ found: false, answer: NOT_FOUND, status: 'ok', repairs: 0 });
    expect(chat.calls).toHaveLength(1);
  });

  it('returns found=null for output that is not even a base Answer', async () => {
    const { deps, requestSink } = setup(['not json', '{"found": "yes"}', '']);
    const res = await answer('q', deps);
    expect(res).toMatchObject({ found: null, answer: null, citations: [], status: 'invalid_output' });
    expect(requestSink.records[0]?.found).toBeNull();
  });

  it('abstains without a model call when nothing was retrieved', async () => {
    const { deps, chat, requestSink } = setup([], []);
    const res = await answer('q', deps);
    expect(res).toMatchObject({
      found: false,
      answer: NOT_FOUND,
      retrievedIds: [],
      context: '',
      status: 'ok',
    });
    expect(chat.calls).toHaveLength(0);
    expect(requestSink.records[0]?.retrievedIds).toEqual([]);
  });

  it('wraps retrieval failures in RetrievalError and writes no request row', async () => {
    const { deps, requestSink } = setup([]);
    const boom = new Error('embed failed');
    deps.retriever = {
      invoke: async () => Promise.reject(boom),
      invokeMany: async () => Promise.reject(boom),
    };
    const err = await answer('q', deps).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(RetrievalError);
    expect((err as RetrievalError).cause).toBe(boom);
    expect(requestSink.records).toHaveLength(0);
  });

  it('retrieves with the rewritten query and answers the original question', async () => {
    const { deps, chat, requestSink } = setup([
      JSON.stringify({ rewritten: 'Bun.env process.env environment variables' }),
      reply(true, 'Use Bun.env.', [11]),
    ]);
    const asked: string[] = [];
    deps.retriever = {
      invoke: async (q: string) => {
        asked.push(q);
        return DOCS;
      },
      invokeMany: async () => {
        throw new Error('invokeMany must not be called without expand');
      },
    };
    deps.rewrite = rewriterFor(chat);
    const res = await answer('How do I read env vars?', deps);
    expect(asked).toEqual(['Bun.env process.env environment variables']);
    expect(chat.calls[1]).toEqual(buildAnswerMessages('How do I read env vars?', formatContext(DOCS)));
    expect(res.rewrite).toEqual({ query: 'Bun.env process.env environment variables', fallback: false });
    expect(res.found).toBe(true);
    expect(requestSink.records[0]?.totalLatencyMs).toBe(res.latencyMs);
  });

  it('retrieves with the question when the rewrite falls back', async () => {
    const { deps, chat } = setup(['{bad', '{bad', '{bad', reply(true, 'Use Bun.env.', [11])]);
    const asked: string[] = [];
    deps.retriever = {
      invoke: async (q: string) => {
        asked.push(q);
        return DOCS;
      },
      invokeMany: async () => {
        throw new Error('invokeMany must not be called without expand');
      },
    };
    deps.rewrite = rewriterFor(chat);
    const res = await answer('How do I read env vars?', deps);
    expect(asked).toEqual(['How do I read env vars?']);
    expect(res.rewrite).toEqual({ query: 'How do I read env vars?', fallback: true });
  });

  it('records no rewrite without the stage', async () => {
    const { deps } = setup([reply(true, 'Use Bun.env.', [11])]);
    expect((await answer('q', deps)).rewrite).toBeNull();
  });

  it('propagates a rewrite provider error as ChatStageError and writes no request row', async () => {
    const boom = new Error('chat down');
    const { deps, chat, requestSink } = setup([]);
    deps.rewrite = rewriterFor(new FakeChatProvider([boom]));
    const err = await answer('q', deps).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ChatStageError);
    expect(err).toMatchObject({ stage: 'rewrite' });
    expect(chat.calls).toHaveLength(0);
    expect(requestSink.records).toHaveLength(0);
  });
});

describe('answer with rerank', () => {
  it('reranks the candidates against the question and answers from the kept top k', async () => {
    const { deps, chat, requestSink } = setup([
      JSON.stringify({
        ranking: [
          { chunk_id: 12, score: 9 },
          { chunk_id: 11, score: 2 },
        ],
      }),
      reply(true, 'Bun reads .env files.', [12]),
    ]);
    deps.rerank = { reranker: rerankerFor(chat), k: 1 };
    const res = await answer('How do I read env vars?', deps);
    expect(chat.calls[0]).toEqual(
      buildRerankMessages('How do I read env vars?', DOCS.map(documentCandidate)),
    );
    const [kept] = DOCS.slice(1);
    expect(chat.calls[1]).toEqual(
      buildAnswerMessages('How do I read env vars?', formatContext(kept ? [kept] : [])),
    );
    expect(res.retrievedIds).toEqual([12]);
    expect(res.rerank).toEqual({
      candidates: [11, 12],
      order: [12, 11],
      scores: [9, 2],
      unknown: 0,
      duplicates: 0,
      missing: 0,
      fallback: false,
    });
    expect(requestSink.records[0]?.retrievedIds).toEqual([12]);
  });

  it('keeps the search order cut to k when the rerank falls back', async () => {
    const { deps, chat } = setup(['{bad', '{bad', '{bad', reply(true, 'Use Bun.env.', [11])]);
    deps.rerank = { reranker: rerankerFor(chat), k: 1 };
    const res = await answer('How do I read env vars?', deps);
    expect(res.retrievedIds).toEqual([11]);
    expect(res.rerank).toMatchObject({ order: [11, 12], fallback: true });
  });

  it('propagates a rerank provider error as ChatStageError, not as RetrievalError', async () => {
    const { deps } = setup([]);
    deps.rerank = { reranker: rerankerFor(new FakeChatProvider([new Error('chat down')])), k: 1 };
    const err = await answer('q', deps).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ChatStageError);
    expect(err).toMatchObject({ stage: 'rerank' });
  });

  it('records no rerank without the stage', async () => {
    const { deps } = setup([reply(true, 'Use Bun.env.', [11])]);
    expect((await answer('q', deps)).rerank).toBeNull();
  });
});

describe('buildAnswerDeps with rerank', () => {
  // postgres() connects lazily; nothing here runs a query.
  const sql = postgres('postgres://localhost:1/unused');
  const config = loadConfig({ CHAT_PROVIDER: 'fake', EMBED_PROVIDER: 'fake' });
  const build = (rerank: 'off' | 'on', k: number) =>
    buildAnswerDeps(sql, config, {
      chunkConfig: 'md-1024-128',
      k,
      runId: 'test-run',
      pipeline: { search: 'vector', rewrite: 'off', expand: 'off', rerank },
    });
  const retrieverK = (deps: AnswerDeps): number => {
    if (!(deps.retriever instanceof PgRetriever)) {
      throw new Error('buildAnswerDeps must build a PgRetriever');
    }
    return deps.retriever.k;
  };

  it('retrieves the rerank candidates and keeps k', () => {
    const deps = build('on', 5);
    expect(retrieverK(deps)).toBe(RERANK_CANDIDATES);
    expect(deps.rerank?.k).toBe(5);
    expect(deps.pipeline).toBe('vector+rerank');
  });

  it('retrieves k without rerank and wires no reranker', () => {
    const deps = build('off', 5);
    expect(retrieverK(deps)).toBe(5);
    expect(deps.rerank).toBeUndefined();
  });
});

describe('answer with expand', () => {
  it('retrieves with the question and its expansions and answers the original question', async () => {
    const { deps, chat, requestSink } = setup([
      JSON.stringify({ queries: ['Bun.env environment variables', 'read .env file'] }),
      reply(true, 'Use Bun.env.', [11]),
    ]);
    const asked: (readonly string[])[] = [];
    deps.retriever = {
      invoke: async () => {
        throw new Error('invoke must not be called with expand');
      },
      invokeMany: async (qs) => {
        asked.push(qs);
        return DOCS;
      },
    };
    deps.expand = expanderFor(chat);
    const res = await answer('How do I read env vars?', deps);
    expect(asked).toEqual([['How do I read env vars?', 'Bun.env environment variables', 'read .env file']]);
    expect(chat.calls[1]).toEqual(buildAnswerMessages('How do I read env vars?', formatContext(DOCS)));
    expect(res.expand).toEqual({
      queries: ['Bun.env environment variables', 'read .env file'],
      dropped: 0,
      fallback: false,
    });
    expect(res.rewrite).toBeNull();
    expect(requestSink.records[0]?.retrievedIds).toEqual([11, 12]);
  });

  it('retrieves with the question alone when the expansion falls back', async () => {
    const { deps, chat } = setup(['{bad', '{bad', '{bad', reply(true, 'Use Bun.env.', [11])]);
    const asked: (readonly string[])[] = [];
    deps.retriever = {
      invoke: async () => DOCS,
      invokeMany: async (qs) => {
        asked.push(qs);
        return DOCS;
      },
    };
    deps.expand = expanderFor(chat);
    const res = await answer('How do I read env vars?', deps);
    expect(asked).toEqual([['How do I read env vars?']]);
    expect(res.expand).toEqual({ queries: [], dropped: 0, fallback: true });
  });

  it('propagates an expand provider error as ChatStageError and wraps a fused retrieval failure', async () => {
    const { deps, requestSink } = setup([]);
    deps.expand = expanderFor(new FakeChatProvider([new Error('chat down')]));
    expect(await answer('q', deps).catch((e: unknown) => e)).toBeInstanceOf(ChatStageError);
    expect(await answer('q', deps).catch((e: unknown) => e)).toMatchObject({ stage: 'expand' });

    const boom = new Error('embed failed');
    deps.expand = expanderFor(new FakeChatProvider([JSON.stringify({ queries: ['x'] })]));
    deps.retriever = { invoke: async () => DOCS, invokeMany: async () => Promise.reject(boom) };
    const err = await answer('q', deps).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(RetrievalError);
    expect((err as RetrievalError).cause).toBe(boom);
    expect(requestSink.records).toHaveLength(0);
  });

  it('names the chat model for an expand ChatStageError 404', () => {
    const config = loadConfig({});
    const notFound = APIError.generate(404, undefined, 'model not found', new Headers());
    expect(describeAnswerError(new ChatStageError('expand', notFound), config, false)).toContain(
      `ollama pull ${config.chatModel}`,
    );
  });
});

describe('ask CLI output', () => {
  const config = loadConfig({});

  it('names the embedding model for a wrapped retrieval 404 and the chat model otherwise', () => {
    const notFound = APIError.generate(404, undefined, 'model not found', new Headers());
    expect(describeAnswerError(new RetrievalError(notFound), config, false)).toContain(
      `ollama pull ${config.embedModel}`,
    );
    expect(describeAnswerError(notFound, config, false)).toContain(`ollama pull ${config.chatModel}`);
    expect(describeAnswerError(new ChatStageError('rewrite', notFound), config, false)).toContain(
      `ollama pull ${config.chatModel}`,
    );
  });

  it('reports Ollama down for a wrapped connection error', () => {
    const down = new APIConnectionError({ message: 'Connection error.' });
    expect(describeAnswerError(new RetrievalError(down), config, false)).toContain('Ollama not reachable');
  });

  it('prints citations, the found flag and warnings', () => {
    const { out, warnings } = formatAnswer({
      requestId: 'r1',
      found: true,
      rewrite: null,
      expand: null,
      rerank: null,
      answer: 'Use Bun.env.',
      citations: [{ chunkId: 12, sourcePath: 'runtime/env.md', headingPath: '' }],
      droppedCitations: [999],
      retrievedIds: [11, 12],
      context: '',
      status: 'invalid_output',
      repairs: 2,
      latencyMs: 5,
    });
    expect(out).toBe('Use Bun.env.\n\nCitations:\n  [chunk:12] runtime/env.md\n\nfound=true  request=r1');
    expect(warnings).toEqual(['dropped citations outside the context: 999']);
  });
});
