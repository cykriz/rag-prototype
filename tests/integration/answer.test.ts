import { join } from 'node:path';
import type { Sql } from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';

import { loadConfig } from '../../src/config';
import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { parseChunkConfigKey } from '../../src/ingest/chunk';
import { ingest } from '../../src/ingest/ingest';
import {
  FAKE_CHAT_MODEL,
  FAKE_JUDGE_MODEL,
  FakeChatProvider,
  FakeEmbeddingProvider,
} from '../../src/llm/fake';
import { LoggedChatProvider, PostgresLlmCallSink } from '../../src/obs/logger';
import { answer, buildAnswerDeps } from '../../src/rag/answer';
import { MAX_K } from '../../src/retrieval/vector';

const FIXTURES = join(import.meta.dirname, '../fixtures/docs');
const CONFIG = loadConfig({ CHAT_PROVIDER: 'fake', EMBED_PROVIDER: 'fake' });
const CHUNK_CONFIG = parseChunkConfigKey('md-64-0');
const FREE = { inputPerMTok: 0, outputPerMTok: 0 };

let sql: Sql;
let chunkIds: number[];

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  await migrate(sql);
  await ensureEmbeddingTables(sql);
  await sql`TRUNCATE documents, chunk_configs CASCADE`;
  await ingest({
    sql,
    root: FIXTURES,
    chunkConfig: CHUNK_CONFIG,
    embedder: new FakeEmbeddingProvider(CONFIG.embedModel, CONFIG.embedding.dim),
    embedding: CONFIG.embedding,
    corpusVersion: 'fixture-v1',
    warn: () => {},
  });
  const rows = await sql<{ id: string }[]>`SELECT id FROM chunks WHERE chunk_config = ${CHUNK_CONFIG.key}`;
  chunkIds = rows.map((r) => Number(r.id));
});

afterAll(async () => {
  await sql?.end();
});

/**
 * buildAnswerDeps with a scripted chat on a sink carrying the same run and request ID; k covers every fixture
 * chunk, so any fixture chunk ID is inside the context.
 */
function scriptedDeps(runId: string, script: string[], questionId?: string) {
  expect(chunkIds.length).toBeLessThanOrEqual(MAX_K);
  const deps = buildAnswerDeps(sql, CONFIG, {
    chunkConfig: CHUNK_CONFIG.key,
    k: chunkIds.length,
    runId,
    pipeline: { search: 'vector', rewrite: 'off', expand: 'off' },
    ...(questionId !== undefined && { questionId }),
  });
  deps.chat = new LoggedChatProvider(new FakeChatProvider(script), {
    sink: new PostgresLlmCallSink(sql, { runId, requestId: deps.requestId }),
    provider: 'fake',
    chatModel: FAKE_CHAT_MODEL,
    judgeModel: FAKE_JUDGE_MODEL,
    chatPrice: FREE,
    judgePrice: FREE,
  });
  return deps;
}

describe('answer with rag_requests logging', () => {
  it('writes one rag_requests row and links the embed and generate calls by request_id', async () => {
    const runId = `test-answer-${crypto.randomUUID()}`;
    const cited = chunkIds[0]!;
    const deps = scriptedDeps(runId, [JSON.stringify({ found: true, answer: 'a', citations: [cited] })]);

    const res = await answer('how do plugins work', deps);

    const [req] = await sql`
      SELECT request_id, run_id, question_id, pipeline, total_latency_ms, retrieved_ids, found
      FROM rag_requests WHERE run_id = ${runId}`;
    expect(req).toEqual({
      request_id: deps.requestId,
      run_id: runId,
      question_id: null,
      pipeline: 'baseline',
      total_latency_ms: res.latencyMs,
      retrieved_ids: res.retrievedIds.map(String),
      found: true,
    });
    expect(res.retrievedIds).toHaveLength(chunkIds.length);

    const calls =
      await sql`SELECT request_id, operation, status FROM llm_calls WHERE run_id = ${runId} ORDER BY id`;
    expect(calls.map((c) => [c.request_id, c.operation, c.status])).toEqual([
      [deps.requestId, 'embed', 'ok'],
      [deps.requestId, 'generate', 'ok'],
    ]);
  });

  it('logs the generate call as invalid_output when invented citations survive the repairs', async () => {
    const runId = `test-answer-${crypto.randomUUID()}`;
    const invented = JSON.stringify({ found: true, answer: 'a', citations: [chunkIds[0], -1] });
    const deps = scriptedDeps(runId, [invented, invented, invented]);

    const res = await answer('how do plugins work', deps);

    expect(res.droppedCitations).toEqual([-1]);
    const calls = await sql`SELECT operation, status FROM llm_calls WHERE run_id = ${runId} ORDER BY id`;
    expect(calls.map((c) => [c.operation, c.status])).toEqual([
      ['embed', 'ok'],
      ['generate', 'invalid_output'],
    ]);
    const [req] = await sql`SELECT found FROM rag_requests WHERE run_id = ${runId}`;
    expect(req?.found).toBe(true);
  });

  it('stores the golden question ID passed through buildAnswerDeps', async () => {
    const runId = `test-answer-${crypto.randomUUID()}`;
    const deps = scriptedDeps(
      runId,
      [JSON.stringify({ found: false, answer: 'NOT_FOUND', citations: [] })],
      'q007',
    );
    await answer('how do plugins work', deps);
    const [req] = await sql`SELECT question_id FROM rag_requests WHERE run_id = ${runId}`;
    expect(req?.question_id).toBe('q007');
  });
});
