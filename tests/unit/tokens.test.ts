import { describe, expect, it } from 'vitest';

import { countMessageTokens, countOutputTokens, countTokens, resolveTokens } from '../../src/llm/tokens';

describe('countTokens', () => {
  it('counts with cl100k_base', () => {
    expect(countTokens('hello world')).toBe(2);
    expect(countTokens('')).toBe(0);
  });

  it('counts special-token strings as plain text instead of throwing', () => {
    expect(countTokens('a <|endoftext|> b')).toBeGreaterThan(3);
  });

  it('sums message contents and assistant tool calls', () => {
    const call = { id: '1', name: 'search_docs', arguments: '{"q":"bun"}' };
    const n = countMessageTokens([
      { role: 'user', content: 'hello world' },
      { role: 'assistant', content: null, toolCalls: [call] },
    ]);
    expect(n).toBe(2 + countTokens('search_docs') + countTokens('{"q":"bun"}'));
    expect(countOutputTokens(null, [call])).toBe(n - 2);
  });
});

describe('resolveTokens', () => {
  it('prefers provider usage over the estimate', () => {
    let estimated = false;
    const res = resolveTokens({ inputTokens: 7, outputTokens: 3 }, () => {
      estimated = true;
      return { inputTokens: 0, outputTokens: 0 };
    });
    expect(res).toEqual({ inputTokens: 7, outputTokens: 3, tokenSource: 'usage' });
    expect(estimated).toBe(false);
  });

  it('falls back to the estimate without usage', () => {
    const res = resolveTokens(undefined, () => ({ inputTokens: 4, outputTokens: 1 }));
    expect(res).toEqual({ inputTokens: 4, outputTokens: 1, tokenSource: 'tiktoken_estimate' });
  });
});
