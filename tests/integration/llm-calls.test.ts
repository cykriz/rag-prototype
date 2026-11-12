import { randomUUID } from 'node:crypto';
import type { Sql } from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import { z } from 'zod';

import { loadConfig } from '../../src/config';
import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { FAKE_CHAT_MODEL, FAKE_JUDGE_MODEL, FakeChatProvider } from '../../src/llm/fake';
import { createChatProvider, createEmbeddingProvider } from '../../src/llm/registry';
import { LoggedChatProvider, PostgresLlmCallSink } from '../../src/obs/logger';

const FAKE_CONFIG = loadConfig({ CHAT_PROVIDER: 'fake', EMBED_PROVIDER: 'fake' });
const FREE = { inputPerMTok: 0, outputPerMTok: 0 };
const REQ = { messages: [{ role: 'user' as const, content: 'hello world' }] };
const Answer = z.object({ answer: z.string() });

let sql: Sql;

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  await migrate(sql);
  await ensureEmbeddingTables(sql);
});

afterAll(async () => {
  await sql?.end();
});

/** A fresh run per test, so rows from other tests or runs never match. */
function newRun() {
  const ctx = { runId: `test-${randomUUID()}`, requestId: randomUUID() };
  return { ctx, sink: new PostgresLlmCallSink(sql, ctx) };
}

function scriptedChat(sink: PostgresLlmCallSink, script: ConstructorParameters<typeof FakeChatProvider>[0]) {
  return new LoggedChatProvider(new FakeChatProvider(script), {
    sink,
    provider: 'fake',
    chatModel: FAKE_CHAT_MODEL,
    judgeModel: FAKE_JUDGE_MODEL,
    chatPrice: FREE,
    judgePrice: FREE,
  });
}

async function rowsOf(runId: string) {
  return sql`
    SELECT request_id, operation, provider, model, input_tokens, output_tokens, token_source, latency_ms,
           cost_usd::float8 AS cost_usd, attempts, status
    FROM llm_calls WHERE run_id = ${runId} ORDER BY id`;
}

describe('PostgresLlmCallSink', () => {
  it('writes one row per call for generate, generateObject, judge and embed', async () => {
    const { ctx, sink } = newRun();
    await createChatProvider(FAKE_CONFIG, sink).generate(REQ);
    const scripted = scriptedChat(sink, ['{"answer":"a"}', '{"answer":"b"}']);
    await scripted.generateObject(REQ, Answer);
    await scripted.judge(REQ, Answer);
    await createEmbeddingProvider(FAKE_CONFIG, sink).embed(['one', 'two'], 'document');

    const rows = await rowsOf(ctx.runId);
    expect(rows.map((r) => [r.operation, r.model, r.status])).toEqual([
      ['generate', FAKE_CHAT_MODEL, 'ok'],
      ['generate', FAKE_CHAT_MODEL, 'ok'],
      ['judge', FAKE_JUDGE_MODEL, 'ok'],
      ['embed', 'nomic-embed-text', 'ok'],
    ]);
    for (const row of rows) {
      expect(row).toMatchObject({
        request_id: ctx.requestId,
        provider: 'fake',
        token_source: 'tiktoken_estimate',
        cost_usd: 0,
        attempts: 1,
      });
      expect(row.input_tokens).toBeGreaterThan(0);
      expect(row.latency_ms).toBeGreaterThanOrEqual(0);
    }
    expect(rows.at(-1)?.output_tokens).toBe(0);
  });

  it('writes one error row when the provider throws', async () => {
    const { ctx, sink } = newRun();
    await expect(scriptedChat(sink, [new Error('boom')]).generate(REQ)).rejects.toThrow('boom');

    const rows = await rowsOf(ctx.runId);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      operation: 'generate',
      status: 'error',
      output_tokens: null,
      cost_usd: 0,
      token_source: 'tiktoken_estimate',
      attempts: 1,
    });
    expect(rows[0]!.input_tokens).toBeGreaterThan(0);
  });

  it('stores a null request_id for calls outside a request', async () => {
    const runId = `test-${randomUUID()}`;
    await createChatProvider(FAKE_CONFIG, new PostgresLlmCallSink(sql, { runId })).generate(REQ);
    expect((await rowsOf(runId)).map((r) => r.request_id)).toEqual([null]);
  });
});
