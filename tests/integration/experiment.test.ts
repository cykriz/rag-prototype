// bun run experiment in process against the test database with fake providers: pre-flight and result files.
// The fake chat provider is created inside the registry; the mock below gives every instance without a script one
// shared queue, so a test can script an answers run end to end (agent turns, then the judge) through the real
// logging wrapper.
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import pgvector from 'pgvector';
import type { Sql } from 'postgres';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, inject, it, vi } from 'vitest';
import { main } from '../../src/cli/experiment';
import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { parseChunkConfigKey } from '../../src/ingest/chunk';
import { sha256 } from '../../src/ingest/hash';
import { fakeEmbedding, type ScriptedReply } from '../../src/llm/fake';

const chatScript = vi.hoisted((): ScriptedReply[] => []);
vi.mock('../../src/llm/fake', async (importOriginal) => {
  const original = await importOriginal<typeof import('../../src/llm/fake')>();
  class SharedScriptChatProvider extends original.FakeChatProvider {
    constructor(script?: ScriptedReply[]) {
      super(script ?? chatScript);
    }
  }
  return { ...original, FakeChatProvider: SharedScriptChatProvider };
});

const CORPUS = 'corpus-sha-experiment';
const SOURCE = 'guides/widget.md';
const TEXT = 'Start the widget with `widget start` on example.com.';

let sql: Sql;
let dir: string;
const errors: string[] = [];

/** One document with one chunk in `config` and its fake nomic embedding. */
async function ingest(config: string): Promise<void> {
  const { chunkSize, chunkOverlap, splitter } = parseChunkConfigKey(config);
  await sql`INSERT INTO chunk_configs (key, chunk_size, chunk_overlap, splitter)
            VALUES (${config}, ${chunkSize}, ${chunkOverlap}, ${splitter}) ON CONFLICT (key) DO NOTHING`;
  const [doc] = await sql<{ id: string }[]>`
    INSERT INTO documents (source_path, title, content_hash, corpus_version)
    VALUES (${SOURCE}, 'Synthetic', ${sha256(SOURCE)}, ${CORPUS})
    ON CONFLICT (source_path) DO UPDATE SET corpus_version = EXCLUDED.corpus_version RETURNING id`;
  const [chunk] = await sql<{ id: string }[]>`
    INSERT INTO chunks (document_id, chunk_config, ordinal, content, content_hash, token_count)
    VALUES (${doc!.id}, ${config}, 0, ${TEXT}, ${sha256(TEXT)}, 10) RETURNING id`;
  await sql`
    INSERT INTO emb_nomic_embed_text (chunk_id, chunk_config, content_hash, embedding)
    VALUES (${chunk!.id}, ${config}, ${sha256(TEXT)}, ${pgvector.toSql(fakeEmbedding(TEXT, 768))})`;
}

async function experimentFile(chunkConfigs: string[]): Promise<string> {
  const path = join(dir, 'exp-chunk-size.json');
  await writeFile(
    path,
    JSON.stringify({
      id: 'exp-chunk-size',
      base: { embed: 'nomic-embed-text', k: 1, search: 'vector', split: 'test' },
      vary: { chunkConfig: chunkConfigs },
      evals: ['retrieval'],
    }),
  );
  return path;
}

const run = (file: string) =>
  main([file, '--dataset', join(dir, 'golden.jsonl'), '--out', join(dir, 'results')]);

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  await migrate(sql);
  await ensureEmbeddingTables(sql);
  await sql`TRUNCATE documents, chunk_configs CASCADE`;
  await ingest('md-512-64');
  await ingest('md-1024-128');
});

afterAll(async () => {
  await sql?.end();
});

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'experiment-'));
  const entry = {
    id: 'q001',
    split: 'test',
    category: 'direct_fact',
    question: TEXT,
    answerable: true,
    evidence: [{ source: SOURCE, span: 'widget start on example.com' }],
    keywords: ['widget'],
    reference_answer: 'Run widget start.',
  };
  await writeFile(join(dir, 'golden.jsonl'), `${JSON.stringify(entry)}\n`);
  vi.stubEnv('DATABASE_URL', inject('databaseUrl'));
  vi.stubEnv('EMBED_PROVIDER', 'fake');
  vi.stubEnv('CHAT_PROVIDER', 'fake');
  vi.stubEnv('CORPUS_SHA', CORPUS);
  errors.length = 0;
  chatScript.length = 0;
  vi.spyOn(console, 'error').mockImplementation((msg: string) => {
    errors.push(msg);
  });
  vi.spyOn(console, 'log').mockImplementation(() => {});
});

afterEach(async () => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  await rm(dir, { recursive: true, force: true });
});

describe('bun run experiment', () => {
  it('pre-flight names every variant without an ingest and runs nothing', async () => {
    const file = await experimentFile(['md-256-32', 'md-512-64', 'md-2048-256']);
    expect(await run(file)).toBe(1);
    const message = errors.join('\n');
    expect(message).toMatch(/pre-flight failed, nothing was run/);
    expect(message).toMatch(
      /No embeddings for md-256-32 \/ nomic-embed-text: run bun run ingest --chunk md-256-32/,
    );
    expect(message).toMatch(/No embeddings for md-2048-256/);
    expect(message).not.toMatch(/md-512-64/);
    await expect(readdir(join(dir, 'results'))).rejects.toThrow();
  });

  it('writes one result file per variant with the variant in its name and header', async () => {
    const file = await experimentFile(['md-512-64', 'md-1024-128']);
    expect(await run(file)).toBe(0);
    // Sorted without the timestamp prefix: the two runs can fall into different seconds.
    const afterStamp = (name: string) => name.slice(name.indexOf('_'));
    const names = (await readdir(join(dir, 'results'))).sort((a, b) =>
      afterStamp(a).localeCompare(afterStamp(b)),
    );
    expect(names).toHaveLength(2);
    expect(names[0]).toMatch(/_exp-chunk-size_md-1024-128_retrieval\.json$/);
    expect(names[1]).toMatch(/_exp-chunk-size_md-512-64_retrieval\.json$/);
    for (const name of names) {
      const result = JSON.parse(await readFile(join(dir, 'results', name), 'utf8'));
      expect(result).toMatchObject({
        experimentId: 'exp-chunk-size',
        evalType: 'retrieval',
        variedKey: 'chunkConfig',
      });
      expect(name).toContain(`_${result.config.chunkConfig}_`);
      expect(result.aggregate.all.recall).toBe(1);
    }
  });
});

describe('bun run experiment with the agent pipeline (M19)', () => {
  it('runs the agent for every question and records its trace, counts and agent_step calls', async () => {
    const [chunk] = await sql<{ id: string }[]>`SELECT id FROM chunks WHERE chunk_config = 'md-1024-128'`;
    const chunkId = Number(chunk?.id);
    const judgeReply = JSON.stringify({ reasoning: 'ok', accuracy: 5, completeness: 4, relevance: 5 });
    chatScript.push(
      { toolCalls: [{ name: 'search_docs', arguments: { query: 'widget start' } }] },
      JSON.stringify({ found: true, answer: 'Run widget start.', citations: [chunkId] }),
      // First judge pass, then the self-consistency repeat (sampleSize(1) = 1).
      judgeReply,
      judgeReply,
    );
    const path = join(dir, 'exp-agent.json');
    await writeFile(
      path,
      JSON.stringify({
        id: 'exp-agent',
        base: {
          chunkConfig: 'md-1024-128',
          embed: 'nomic-embed-text',
          k: 1,
          search: 'vector',
          split: 'test',
        },
        vary: { pipeline: ['agent'] },
        evals: ['answers'],
      }),
    );
    expect(await run(path)).toBe(0);
    expect(chatScript).toEqual([]);
    const names = await readdir(join(dir, 'results'));
    expect(names).toEqual([expect.stringMatching(/_exp-agent_agent_answers\.json$/)]);
    const result = JSON.parse(await readFile(join(dir, 'results', names[0] ?? ''), 'utf8'));
    expect(result).toMatchObject({
      schemaVersion: 6,
      config: { pipeline: 'agent', rerank: 'off' },
      variedKey: 'pipeline',
      counts: { questions: 1, judged: 1, agentToolCalls: 1, agentStoppedAtLimit: 0, agentAnswerFallbacks: 0 },
    });
    expect(result.questions[0]).toMatchObject({
      found: true,
      citations: [chunkId],
      retrievedIds: [chunkId],
      agent: {
        trace: [{ query: 'widget start', kRequested: null, k: 1, chunkIds: [chunkId], error: null }],
        toolCalls: 1,
        skippedToolCalls: 0,
        stoppedAtLimit: false,
        answerFallback: false,
      },
      judge: { status: 'ok', accuracy: 5 },
    });
    const ops = Object.fromEntries(
      result.kpis.calls.map((c: { operation: string; calls: number }) => [c.operation, c.calls]),
    );
    expect(ops).toEqual({ agent_step: 2, embed: 1, judge: 2 });
    const [req] = await sql`SELECT pipeline FROM rag_requests WHERE run_id = ${result.runId}`;
    expect(req?.pipeline).toBe('agent');
  });
});
