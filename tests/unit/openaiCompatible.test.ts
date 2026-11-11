// Request mapping of the OpenAI-compatible adapter, verified with a stub fetch (docs/PLAN.md §9). No network.
import { describe, expect, it } from 'vitest';
import { z } from 'zod';

import { models } from '../../src/config';
import {
  OpenAICompatibleChatProvider,
  OpenAICompatibleEmbeddingProvider,
} from '../../src/llm/openaiCompatible';

const BASE_URL = 'http://localhost:11434/v1';
const noWait = { sleep: async () => {}, random: () => 0 };

interface Captured {
  url: string;
  body: Record<string, unknown>;
}

function stubFetch(responses: Array<{ status: number; body: unknown }>) {
  const requests: Captured[] = [];
  const fetchFn = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    requests.push({ url, body: JSON.parse(String(init?.body ?? '{}')) });
    const next = responses[Math.min(requests.length - 1, responses.length - 1)]!;
    return new Response(JSON.stringify(next.body), {
      status: next.status,
      headers: { 'content-type': 'application/json' },
    });
  }) as typeof fetch;
  return { fetchFn, requests };
}

const chatBody = (message: Record<string, unknown>) => ({
  id: 'c1',
  object: 'chat.completion',
  created: 0,
  model: 'qwen2.5:7b',
  choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', ...message } }],
  usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
});

const embedBody = (dim: number, n = 1) => ({
  object: 'list',
  model: 'nomic-embed-text',
  data: Array.from({ length: n }, (_, index) => ({
    object: 'embedding',
    index,
    embedding: Array(dim).fill(0.1),
  })),
  usage: { prompt_tokens: 4, total_tokens: 4 },
});

const nomic = models.embeddings['nomic-embed-text']!;

function chat(responses: Array<{ status: number; body: unknown }>) {
  const stub = stubFetch(responses);
  const provider = new OpenAICompatibleChatProvider({
    baseURL: BASE_URL,
    chatModel: 'qwen2.5:7b',
    judgeModel: 'gemma3:12b',
    fetch: stub.fetchFn,
    retry: noWait,
  });
  return { provider, requests: stub.requests };
}

function embedder(responses: Array<{ status: number; body: unknown }>) {
  const stub = stubFetch(responses);
  const provider = new OpenAICompatibleEmbeddingProvider('nomic-embed-text', nomic, {
    baseURL: BASE_URL,
    fetch: stub.fetchFn,
    retry: noWait,
  });
  return { provider, requests: stub.requests };
}

describe('OpenAICompatibleEmbeddingProvider', () => {
  it('applies the nomic document prefix and requests float encoding', async () => {
    const { provider, requests } = embedder([{ status: 200, body: embedBody(768, 2) }]);
    const res = await provider.embed(['a', 'b'], 'document');
    expect(requests[0]!.url).toBe(`${BASE_URL}/embeddings`);
    expect(requests[0]!.body).toMatchObject({
      model: 'nomic-embed-text',
      input: ['search_document: a', 'search_document: b'],
      encoding_format: 'float',
    });
    expect(res.vectors).toHaveLength(2);
    expect(res.usage).toEqual({ inputTokens: 4, outputTokens: 0 });
  });

  it('applies the nomic query prefix', async () => {
    const { provider, requests } = embedder([{ status: 200, body: embedBody(768) }]);
    await provider.embed(['how'], 'query');
    expect(requests[0]!.body.input).toEqual(['search_query: how']);
  });

  it('throws when the returned vector has the wrong dimension', async () => {
    const { provider } = embedder([{ status: 200, body: embedBody(1024) }]);
    await expect(provider.embed(['a'], 'document')).rejects.toThrow('expected dim 768, got 1024');
  });
});

describe('OpenAICompatibleChatProvider', () => {
  it('sends a json_schema response_format and validates the result', async () => {
    const { provider, requests } = chat([{ status: 200, body: chatBody({ content: '{"ok":true}' }) }]);
    const res = await provider.judge(
      { messages: [{ role: 'user', content: 'q' }], temperature: 0 },
      z.object({ ok: z.boolean() }),
    );
    expect(res).toMatchObject({ status: 'ok', value: { ok: true }, attempts: 1 });
    expect(requests[0]!.body).toMatchObject({
      model: 'gemma3:12b',
      temperature: 0,
      response_format: { type: 'json_schema', json_schema: { name: 'output' } },
    });
  });

  it('retries 503 with its own retry only (SDK retries off): 3 fetch calls', async () => {
    const error = { status: 503, body: { error: { message: 'busy' } } };
    const { provider, requests } = chat([error, error, { status: 200, body: chatBody({ content: 'hi' }) }]);
    const res = await provider.generate({ messages: [{ role: 'user', content: 'hello' }] });
    expect(requests).toHaveLength(3);
    expect(res).toMatchObject({ text: 'hi', attempts: 3, usage: { inputTokens: 7, outputTokens: 3 } });
  });

  it('maps tools and tool messages to the request and tool_calls back', async () => {
    const toolCall = {
      id: 't1',
      type: 'function',
      function: { name: 'search_docs', arguments: '{"query":"env"}' },
    };
    const { provider, requests } = chat([
      { status: 200, body: chatBody({ content: null, tool_calls: [toolCall] }) },
    ]);
    const res = await provider.generate({
      messages: [
        { role: 'user', content: 'q' },
        {
          role: 'assistant',
          content: null,
          toolCalls: [{ id: 't0', name: 'search_docs', arguments: '{"query":"x"}' }],
        },
        { role: 'tool', toolCallId: 't0', content: 'result' },
      ],
      tools: [{ name: 'search_docs', description: 'Search', parameters: { type: 'object', properties: {} } }],
    });
    expect(requests[0]!.body).toMatchObject({
      messages: [
        { role: 'user', content: 'q' },
        {
          role: 'assistant',
          content: null,
          tool_calls: [
            { id: 't0', type: 'function', function: { name: 'search_docs', arguments: '{"query":"x"}' } },
          ],
        },
        { role: 'tool', tool_call_id: 't0', content: 'result' },
      ],
      tools: [{ type: 'function', function: { name: 'search_docs', description: 'Search' } }],
    });
    expect(res.toolCalls).toEqual([{ id: 't1', name: 'search_docs', arguments: '{"query":"env"}' }]);
    expect(res.text).toBeNull();
  });
});
