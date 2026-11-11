import { APIConnectionError, APIError } from 'openai';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';

import { describeError } from '../../src/cli/askRaw';
import { loadConfig } from '../../src/config';
import { FAKE_JUDGE_MODEL, FakeChatProvider, FakeEmbeddingProvider, fakeEmbedding } from '../../src/llm/fake';
import {
  OpenAICompatibleChatProvider,
  OpenAICompatibleEmbeddingProvider,
} from '../../src/llm/openaiCompatible';
import { createChatProvider, createEmbeddingProvider } from '../../src/llm/registry';

describe('loadConfig', () => {
  it('applies defaults for an empty environment', () => {
    expect(loadConfig({})).toEqual({
      chatProvider: 'ollama',
      embedProvider: 'ollama',
      baseURL: 'http://localhost:11434/v1',
      chatModel: 'qwen2.5:7b',
      judgeModel: 'gemma3:12b',
      embedModel: 'nomic-embed-text',
      embedding: { dim: 768, documentPrefix: 'search_document: ', queryPrefix: 'search_query: ' },
    });
  });

  it('resolves the light profile and another embedding model', () => {
    const c = loadConfig({ MODEL_PROFILE: 'light', EMBED_MODEL: 'mxbai-embed-large' });
    expect(c).toMatchObject({ chatModel: 'qwen2.5:3b', judgeModel: 'llama3.1:8b', embedding: { dim: 1024 } });
  });

  it('treats empty strings as unset', () => {
    expect(loadConfig({ CHAT_PROVIDER: '' }).chatProvider).toBe('ollama');
  });

  it.each([
    ['CHAT_PROVIDER', 'openai'],
    ['EMBED_MODEL', 'unknown-model'],
    ['OLLAMA_BASE_URL', 'not a url'],
  ])('rejects invalid %s', (key, value) => {
    expect(() => loadConfig({ [key]: value })).toThrow(new RegExp(`Invalid configuration: ${key}`));
  });
});

describe('registry', () => {
  it('builds the OpenAI-compatible adapters by default', () => {
    const config = loadConfig({});
    expect(createChatProvider(config)).toBeInstanceOf(OpenAICompatibleChatProvider);
    expect(createEmbeddingProvider(config)).toBeInstanceOf(OpenAICompatibleEmbeddingProvider);
  });

  it('switches to the fakes via env only', async () => {
    const config = loadConfig({ CHAT_PROVIDER: 'fake', EMBED_PROVIDER: 'fake' });
    const chat = createChatProvider(config);
    const embed = createEmbeddingProvider(config);
    expect(chat).toBeInstanceOf(FakeChatProvider);
    expect(embed).toBeInstanceOf(FakeEmbeddingProvider);
    expect(embed.dim).toBe(768);
    const res = await chat.generate({ messages: [{ role: 'user', content: 'hello' }] });
    expect(res.text).toBe('fake: hello');
  });
});

describe('fake providers', () => {
  it('embeddings are deterministic, unit norm and dim long, without task prefixes', async () => {
    const provider = new FakeEmbeddingProvider('nomic-embed-text', 768);
    const doc = await provider.embed(['same text'], 'document');
    const query = await provider.embed(['same text'], 'query');
    const v = doc.vectors[0]!;
    expect(v).toHaveLength(768);
    expect(Math.hypot(...v)).toBeCloseTo(1, 10);
    expect(query.vectors[0]).toEqual(v);
    expect(fakeEmbedding('other text', 768)).not.toEqual(v);
  });

  it('serves generateObject and judge from one scripted queue in call order', async () => {
    const chat = new FakeChatProvider(['{"n":1}', '{"n":2}']);
    const schema = z.object({ n: z.number() });
    const req = { messages: [{ role: 'user' as const, content: 'q' }] };
    expect(await chat.generateObject(req, schema)).toMatchObject({ status: 'ok', value: { n: 1 } });
    expect(await chat.judge(req, schema)).toMatchObject({
      status: 'ok',
      value: { n: 2 },
      model: FAKE_JUDGE_MODEL,
    });
    await expect(chat.generateObject(req, schema)).rejects.toThrow('no scripted response');
  });
});

describe('ask:raw error messages', () => {
  const config = { baseURL: 'http://localhost:11434/v1', chatModel: 'qwen2.5:7b' };

  it('hints at starting Ollama on connection errors', () => {
    expect(describeError(new APIConnectionError({ message: 'Connection error.' }), config)).toContain(
      'start it (ollama serve)',
    );
  });

  it('hints at pulling the model on 404', () => {
    const err = APIError.generate(404, undefined, 'model not found', new Headers());
    expect(describeError(err, config)).toBe(
      'Model call failed with status 404: 404 model not found (is the model pulled? ollama pull qwen2.5:7b)',
    );
  });

  it('prints status and message for other API errors', () => {
    const err = APIError.generate(400, undefined, 'bad request', new Headers());
    expect(describeError(err, config)).toMatch(/^Model call failed with status 400/);
  });
});
