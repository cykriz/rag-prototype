// M18: search_docs argument parsing, fence stripping, scripted tool calls in the fake, trace formatting, prompt.
import { describe, expect, it } from 'vitest';

import { stripFence } from '../../src/agent/graph';
import { parseSearchArgs, SEARCH_DOCS, searchDocsTool } from '../../src/agent/tools';
import { formatAgentTrace } from '../../src/cli/ask';
import { FakeChatProvider } from '../../src/llm/fake';
import { MAX_TOOL_K } from '../../src/rag/pipeline';
import { buildAgentMessages } from '../../src/rag/prompts';
import { Answer } from '../../src/rag/schemas';

const call = (args: unknown, name = SEARCH_DOCS) => ({
  name,
  arguments: typeof args === 'string' ? args : JSON.stringify(args),
});

describe('parseSearchArgs', () => {
  it('uses the default k when k is absent or null', () => {
    expect(parseSearchArgs(call({ query: ' bun test ' }), 5)).toEqual({
      ok: true,
      query: 'bun test',
      kRequested: null,
      k: 5,
    });
    expect(parseSearchArgs(call({ query: 'q', k: null }), 5)).toMatchObject({ ok: true, k: 5 });
  });

  it('clamps an integer k to 1–MAX_TOOL_K and accepts integer strings', () => {
    expect(parseSearchArgs(call({ query: 'q', k: 50 }), 5)).toMatchObject({
      ok: true,
      kRequested: 50,
      k: MAX_TOOL_K,
    });
    expect(parseSearchArgs(call({ query: 'q', k: -3 }), 5)).toMatchObject({ ok: true, kRequested: -3, k: 1 });
    expect(parseSearchArgs(call({ query: 'q', k: '7' }), 5)).toMatchObject({ ok: true, kRequested: 7, k: 7 });
  });

  it('rejects a non-integer k, a blank or over-long query, bad JSON and unknown tools', () => {
    expect(parseSearchArgs(call({ query: 'q', k: 2.5 }), 5)).toMatchObject({
      ok: false,
      error: expect.stringContaining('k:'),
    });
    expect(parseSearchArgs(call({ query: 'q', k: 'many' }), 5)).toMatchObject({ ok: false });
    expect(parseSearchArgs(call({ query: '   ' }), 5)).toMatchObject({
      ok: false,
      error: expect.stringContaining('query'),
    });
    expect(parseSearchArgs(call({ query: 'x'.repeat(501) }), 5)).toMatchObject({ ok: false });
    expect(parseSearchArgs(call({}), 5)).toMatchObject({ ok: false });
    expect(parseSearchArgs(call('{"query":'), 5)).toEqual({
      ok: false,
      error: 'arguments are not valid JSON',
    });
    expect(parseSearchArgs(call({ query: 'q' }, 'web'), 5)).toMatchObject({
      ok: false,
      error: expect.stringContaining('unknown tool "web"'),
    });
  });

  it('describes k with its bounds and the default in the tool definition', () => {
    expect(searchDocsTool(5).parameters).toMatchObject({
      required: ['query'],
      properties: {
        k: { minimum: 1, maximum: MAX_TOOL_K, description: expect.stringContaining('default 5') },
      },
    });
  });
});

describe('stripFence', () => {
  it('removes one surrounding code fence and leaves other text alone', () => {
    expect(stripFence('```json\n{"a":1}\n```')).toBe('{"a":1}');
    expect(stripFence('```\n{"a":1}```  ')).toBe('{"a":1}');
    expect(stripFence('{"a":1}')).toBe('{"a":1}');
    expect(stripFence('see ```x```')).toBe('see ```x```');
  });
});

describe('FakeChatProvider tool calls', () => {
  it('returns scripted tool calls from generate with running ids and JSON arguments', async () => {
    const fake = new FakeChatProvider([
      { toolCalls: [{ name: 'search_docs', arguments: { query: 'a' } }] },
      { toolCalls: [{ name: 'search_docs', arguments: '{"query":"b"}' }], text: 'thinking' },
      'done',
    ]);
    const req = { messages: [{ role: 'user' as const, content: 'q' }] };
    expect(await fake.generate(req)).toMatchObject({
      text: null,
      toolCalls: [{ id: 'call_1', name: 'search_docs', arguments: '{"query":"a"}' }],
    });
    expect(await fake.generate(req)).toMatchObject({ text: 'thinking', toolCalls: [{ id: 'call_2' }] });
    expect(await fake.generate(req)).toMatchObject({ text: 'done', toolCalls: [] });
  });

  it('refuses a scripted tool call in generateObject', async () => {
    const fake = new FakeChatProvider([{ toolCalls: [] }]);
    await expect(fake.generateObject({ messages: [] }, Answer)).rejects.toThrow('for generate only');
  });
});

describe('formatAgentTrace', () => {
  it('prints query, k and chunk IDs per call, errors, the limit and the answer fallback', () => {
    const out = formatAgentTrace({
      trace: [
        { query: 'bun test', kRequested: null, k: 5, chunkIds: [12, 34], error: null },
        { query: 'mocks', kRequested: 50, k: 10, chunkIds: [], error: null },
        { query: null, kRequested: null, k: null, chunkIds: [], error: 'arguments are not valid JSON' },
        { query: 'x', kRequested: 2, k: 2, chunkIds: [7], error: null },
      ],
      skippedToolCalls: 1,
      stoppedAtLimit: true,
      answerFallback: true,
    });
    expect(out.split('\n')).toEqual([
      '[1] search_docs("bun test", k=5) → 12, 34',
      '[2] search_docs("mocks", k=10 (requested 50)) → (no results)',
      '[3] search_docs error: arguments are not valid JSON',
      '[4] search_docs("x", k=2) → 7',
      '[limit] stopped after 4 tool calls; 1 more not executed',
      '[answer] structured answer call over the retrieved chunks',
    ]);
  });

  it('says so when the agent did not search', () => {
    expect(
      formatAgentTrace({ trace: [], skippedToolCalls: 0, stoppedAtLimit: false, answerFallback: false }),
    ).toBe('[0] no search');
  });
});

describe('agent prompt', () => {
  it('matches the snapshot', () => {
    expect(buildAgentMessages('How do I run tests?')).toMatchSnapshot();
  });
});
