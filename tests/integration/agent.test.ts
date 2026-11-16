// M18a acceptance: the LangGraph agent with a scripted fake chat against the fixture ingest.
import { join } from 'node:path';
import type { Sql } from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';

import { type AgentDeps, buildAgentDeps, runAgent } from '../../src/agent/graph';
import { loadConfig } from '../../src/config';
import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { parseChunkConfigKey } from '../../src/ingest/chunk';
import { embedText } from '../../src/ingest/hash';
import { ingest } from '../../src/ingest/ingest';
import {
  FAKE_CHAT_MODEL,
  FAKE_JUDGE_MODEL,
  FakeChatProvider,
  FakeEmbeddingProvider,
  type ScriptedReply,
  type ScriptedToolCalls,
} from '../../src/llm/fake';
import type { ChatProvider, GenerateRequest, GenerateResult, ObjectResult } from '../../src/llm/types';
import { LoggedChatProvider, PostgresLlmCallSink } from '../../src/obs/logger';
import { RetrievalError } from '../../src/rag/answer';
import { MAX_TOOL_CALLS, MAX_TOOL_K } from '../../src/rag/pipeline';

const FIXTURES = join(import.meta.dirname, '../fixtures/docs');
const CONFIG = loadConfig({ CHAT_PROVIDER: 'fake', EMBED_PROVIDER: 'fake' });
const CHUNK_CONFIG = parseChunkConfigKey('md-64-0');
const FREE = { inputPerMTok: 0, outputPerMTok: 0 };
const DEFAULT_K = 3;
const ABSTAIN = JSON.stringify({ found: false, answer: 'NOT_FOUND', citations: [] });

let sql: Sql;
/** A fixture chunk and its embedded text: searching that text returns the chunk first (fake embeddings, M8). */
let target: { id: number; query: string };

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
  const rows = await sql<{ id: string; heading_path: string; content: string }[]>`
    SELECT id, heading_path, content FROM chunks WHERE chunk_config = ${CHUNK_CONFIG.key} ORDER BY id`;
  const texts = rows.map((r) => ({
    id: Number(r.id),
    query: embedText({ headingPath: r.heading_path, content: r.content }),
  }));
  const fit = texts.find((t) => t.query.trim() === t.query && t.query.length <= 500);
  if (!fit) {
    throw new Error('no fixture chunk usable as a search_docs query');
  }
  target = fit;
});

afterAll(async () => {
  await sql?.end();
});

/** buildAgentDeps with `chat` wrapped in a LoggedChatProvider on a sink carrying the same run and request ID. */
function agentDeps(runId: string, chat: ChatProvider): AgentDeps {
  const deps = buildAgentDeps(sql, CONFIG, {
    chunkConfig: CHUNK_CONFIG.key,
    k: DEFAULT_K,
    search: 'vector',
    runId,
  });
  deps.chat = new LoggedChatProvider(chat, {
    sink: new PostgresLlmCallSink(sql, { runId, requestId: deps.requestId }),
    provider: 'fake',
    chatModel: FAKE_CHAT_MODEL,
    judgeModel: FAKE_JUDGE_MODEL,
    chatPrice: FREE,
    judgePrice: FREE,
  });
  return deps;
}

const search = (query: string, k?: unknown): ScriptedToolCalls => ({
  toolCalls: [{ name: 'search_docs', arguments: k === undefined ? { query } : { query, k } }],
});

const answerJson = (citations: number[]) =>
  JSON.stringify({ found: true, answer: 'Use bun test.', citations });

async function operations(runId: string): Promise<[string, string][]> {
  const rows = await sql`SELECT operation, status FROM llm_calls WHERE run_id = ${runId} ORDER BY id`;
  return rows.map((r) => [r.operation, r.status]);
}

function scripted(script: ScriptedReply[]) {
  const runId = `test-agent-${crypto.randomUUID()}`;
  const fake = new FakeChatProvider(script);
  return { runId, fake, deps: agentDeps(runId, fake) };
}

/** Asks for one more search on every turn; the answer call abstains. */
class LoopingChat implements ChatProvider {
  turns = 0;
  async generate(_req: GenerateRequest): Promise<GenerateResult> {
    this.turns++;
    const call = {
      id: `loop_${this.turns}`,
      name: 'search_docs',
      arguments: JSON.stringify({ query: 'again' }),
    };
    return { text: null, toolCalls: [call], model: FAKE_CHAT_MODEL, attempts: 1 };
  }
  async generateObject<T>(): Promise<ObjectResult<T>> {
    return { status: 'ok', value: JSON.parse(ABSTAIN) as T, model: FAKE_CHAT_MODEL, attempts: 1, repairs: 0 };
  }
  judge<T>(): Promise<ObjectResult<T>> {
    throw new Error('not used');
  }
}

describe('agent graph', () => {
  it('answers after two tool calls with a valid Answer, logging one agent_step per step', async () => {
    const { runId, fake, deps } = scripted([
      search(target.query, 2),
      search('how do plugins work'),
      answerJson([target.id]),
    ]);

    const res = await runAgent('how do I run tests', deps);

    expect(res).toMatchObject({
      found: true,
      answer: 'Use bun test.',
      citations: [{ chunkId: target.id }],
      droppedCitations: [],
      status: 'ok',
      repairs: 0,
      toolCalls: 2,
      skippedToolCalls: 0,
      stoppedAtLimit: false,
      answerFallback: false,
      rewrite: null,
      expand: null,
      rerank: null,
    });
    expect(res.trace.map((s) => [s.query, s.kRequested, s.k, s.chunkIds.length, s.error])).toEqual([
      [target.query, 2, 2, 2, null],
      ['how do plugins work', null, DEFAULT_K, DEFAULT_K, null],
    ]);
    expect(res.trace[0]?.chunkIds[0]).toBe(target.id);
    // The union of both searches, first-seen order, deduplicated; the judge sees exactly these chunks as context.
    const union = [...new Set(res.trace.flatMap((s) => s.chunkIds))];
    expect(res.retrievedIds).toEqual(union);
    expect(res.context).toContain(`[chunk:${target.id}]`);
    // The second agent turn saw the first tool result.
    expect(fake.calls[1]?.at(-1)).toMatchObject({ role: 'tool', toolCallId: 'call_1' });

    expect(await operations(runId)).toEqual([
      ['agent_step', 'ok'],
      ['embed', 'ok'],
      ['agent_step', 'ok'],
      ['embed', 'ok'],
      ['agent_step', 'ok'],
    ]);
    const [req] =
      await sql`SELECT request_id, pipeline, retrieved_ids, found FROM rag_requests WHERE run_id = ${runId}`;
    expect(req).toEqual({
      request_id: deps.requestId,
      pipeline: 'agent',
      retrieved_ids: union.map(String),
      found: true,
    });
  });

  it('stops an endless tool loop at the limit and forces the answer without tools', async () => {
    const runId = `test-agent-${crypto.randomUUID()}`;
    const chat = new LoopingChat();
    const res = await runAgent('loop', agentDeps(runId, chat));

    expect(chat.turns).toBe(MAX_TOOL_CALLS);
    expect(res).toMatchObject({
      toolCalls: MAX_TOOL_CALLS,
      stoppedAtLimit: true,
      answerFallback: true,
      found: false,
    });
    expect(res.trace).toHaveLength(MAX_TOOL_CALLS);
    const ops = await operations(runId);
    expect(ops.filter(([op]) => op === 'agent_step')).toHaveLength(MAX_TOOL_CALLS + 1);
    expect(ops.filter(([op]) => op === 'embed')).toHaveLength(MAX_TOOL_CALLS);
  });

  it('counts parallel tool calls individually and skips the ones over the limit', async () => {
    const three: ScriptedToolCalls = {
      toolCalls: ['a', 'b', 'c'].map((query) => ({ name: 'search_docs', arguments: { query } })),
    };
    const { deps } = scripted([three, three, ABSTAIN]);
    const res = await runAgent('parallel', deps);
    expect(res).toMatchObject({ toolCalls: MAX_TOOL_CALLS, skippedToolCalls: 2, stoppedAtLimit: true });
  });

  it('clamps an out-of-range k and reports an invalid k or tool back to the model', async () => {
    const { fake, deps } = scripted([
      {
        toolCalls: [
          { name: 'search_docs', arguments: { query: 'plugins', k: 50 } },
          { name: 'search_docs', arguments: { query: 'plugins', k: 0 } },
        ],
      },
      search('plugins', 2.5),
      { toolCalls: [{ name: 'web_search', arguments: { query: 'plugins' } }] },
      ABSTAIN,
    ]);

    const res = await runAgent('k bounds', deps);

    expect(res.trace.map((s) => [s.kRequested, s.k, s.error === null])).toEqual([
      [50, MAX_TOOL_K, true],
      [0, 1, true],
      [null, null, false],
      [null, null, false],
    ]);
    expect(res.trace[0]?.chunkIds.length).toBeGreaterThan(0);
    expect(res.trace[2]?.error).toContain('k');
    expect(res.trace[3]?.error).toContain('unknown tool "web_search"');
    // Errors count toward the limit, so the fourth call ended the loop and the model was not asked again.
    expect(res).toMatchObject({ toolCalls: MAX_TOOL_CALLS, stoppedAtLimit: true, answerFallback: true });
    expect(fake.calls[2]?.at(-1)).toMatchObject({ role: 'tool', content: expect.stringMatching(/^error: /) });
  });

  it('repairs a final answer citing IDs outside the retrieved chunks through the answer fallback', async () => {
    const { runId, fake, deps } = scripted([
      search(target.query),
      answerJson([999_999]),
      answerJson([target.id]),
    ]);

    const res = await runAgent('foreign', deps);

    expect(res).toMatchObject({
      found: true,
      citations: [{ chunkId: target.id }],
      droppedCitations: [],
      status: 'ok',
      answerFallback: true,
    });
    // The fallback prompt shows the union as context, not the agent conversation.
    expect(fake.calls[2]?.[1]?.content).toContain(`[chunk:${target.id}]`);
    expect(await operations(runId)).toEqual([
      ['agent_step', 'ok'],
      ['embed', 'ok'],
      ['agent_step', 'ok'],
      ['agent_step', 'ok'],
    ]);
  });

  it('drops foreign citations that survive the repairs and logs invalid_output', async () => {
    const foreign = answerJson([target.id, 999_999]);
    const { runId, deps } = scripted([search(target.query), foreign, foreign, foreign, foreign]);

    const res = await runAgent('foreign twice', deps);

    expect(res).toMatchObject({
      found: true,
      citations: [{ chunkId: target.id }],
      droppedCitations: [999_999],
      status: 'invalid_output',
      repairs: 2,
    });
    expect((await operations(runId)).at(-1)).toEqual(['agent_step', 'invalid_output']);
  });

  it('returns found=null when the final output stays invalid after the repairs', async () => {
    const { runId, deps } = scripted([search('plugins'), 'not json', 'still not', 'nope', 'never']);

    const res = await runAgent('invalid', deps);

    expect(res).toMatchObject({
      found: null,
      answer: null,
      status: 'invalid_output',
      repairs: 2,
      answerFallback: true,
    });
    const [req] = await sql`SELECT found FROM rag_requests WHERE run_id = ${runId}`;
    expect(req?.found).toBeNull();
  });

  it('accepts a fenced JSON reply and abstains without a call when nothing was searched', async () => {
    const fenced = scripted([search(target.query), `\`\`\`json\n${answerJson([target.id])}\n\`\`\``]);
    expect(await runAgent('fenced', fenced.deps)).toMatchObject({ found: true, answerFallback: false });

    const direct = scripted(['Bun is fast.']);
    const res = await runAgent('no search', direct.deps);
    expect(res).toMatchObject({
      found: false,
      answer: 'NOT_FOUND',
      retrievedIds: [],
      context: '',
      toolCalls: 0,
    });
    expect(await operations(direct.runId)).toEqual([['agent_step', 'ok']]);
  });

  it('wraps a failing search in RetrievalError', async () => {
    const { deps } = scripted([search('plugins')]);
    deps.search = async () => {
      throw new Error('db gone');
    };
    await expect(runAgent('fails', deps)).rejects.toBeInstanceOf(RetrievalError);
  });
});
