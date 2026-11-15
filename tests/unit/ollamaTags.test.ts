import { describe, expect, it } from 'vitest';
import { fetchModelDigest, findDigest, ollamaApiBase } from '../../src/llm/ollamaTags';
import { isConnectionError } from '../../src/llm/retry';

const LISTED = [
  { name: 'nomic-embed-text:latest', digest: 'sha-nomic' },
  { name: 'mxbai-embed-large:v1', digest: 'sha-mxbai' },
];

const stubFetch = (body: unknown, status = 200) => {
  const calls: string[] = [];
  const fn = (async (url: string) => {
    calls.push(url);
    return new Response(JSON.stringify(body), { status });
  }) as unknown as typeof fetch;
  return { fn, calls };
};

describe('ollama tags', () => {
  it('drops the OpenAI-compatible /v1 suffix', () => {
    expect(ollamaApiBase('http://localhost:11434/v1')).toBe('http://localhost:11434');
    expect(ollamaApiBase('http://localhost:11434/v1/')).toBe('http://localhost:11434');
    expect(ollamaApiBase('http://localhost:11434')).toBe('http://localhost:11434');
  });

  it('finds a model by exact name or as :latest', () => {
    expect(findDigest(LISTED, 'nomic-embed-text')).toEqual({
      tag: 'nomic-embed-text:latest',
      digest: 'sha-nomic',
    });
    expect(findDigest(LISTED, 'mxbai-embed-large:v1')?.digest).toBe('sha-mxbai');
    expect(findDigest(LISTED, 'mxbai-embed-large')).toBeUndefined();
  });

  it('fetches /api/tags and names the pull command for a missing model', async () => {
    const { fn, calls } = stubFetch({ models: LISTED });
    await expect(fetchModelDigest('http://localhost:11434/v1', 'nomic-embed-text', fn)).resolves.toEqual({
      tag: 'nomic-embed-text:latest',
      digest: 'sha-nomic',
    });
    expect(calls).toEqual(['http://localhost:11434/api/tags']);
    await expect(fetchModelDigest('http://localhost:11434/v1', 'qwen-missing', fn)).rejects.toThrow(
      /ollama pull qwen-missing/,
    );
  });

  it('reports a non-200 status', async () => {
    const { fn } = stubFetch({}, 500);
    await expect(fetchModelDigest('http://localhost:11434/v1', 'x', fn)).rejects.toThrow(/status 500/);
  });

  it('lets a refused connection through as a connection error', async () => {
    const refused = Object.assign(new TypeError('Unable to connect'), { code: 'ECONNREFUSED' });
    const fn = (async () => {
      throw refused;
    }) as unknown as typeof fetch;
    const err = await fetchModelDigest('http://localhost:11434/v1', 'x', fn).catch((e: unknown) => e);
    expect(isConnectionError(err)).toBe(true);
  });
});
