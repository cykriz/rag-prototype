import { describe, expect, it } from 'vitest';
import { z } from 'zod';

import { generateObject, type RawCompletion } from '../../src/llm/structured';
import type { Message } from '../../src/llm/types';

const Answer = z.object({ found: z.boolean(), answer: z.string(), citations: z.array(z.number().int()) });
const VALID = JSON.stringify({ found: true, answer: 'yes', citations: [1] });
const PROMPT: Message[] = [{ role: 'user', content: 'question' }];

function scripted(replies: string[]) {
  const calls: Message[][] = [];
  const complete = async (messages: Message[]): Promise<RawCompletion> => {
    calls.push(messages);
    const content = replies[calls.length - 1] ?? null;
    return { content, model: 'm', attempts: 1, usage: { inputTokens: 10, outputTokens: 5 } };
  };
  return { complete, calls };
}

describe('generateObject', () => {
  it('returns the value without repair when the first output is valid', async () => {
    const { complete, calls } = scripted([VALID]);
    const res = await generateObject(complete, PROMPT, Answer);
    expect(res).toMatchObject({ status: 'ok', value: { found: true }, repairs: 0, attempts: 1 });
    expect(calls).toHaveLength(1);
  });

  it('repairs invalid JSON and passes the error back to the model', async () => {
    const { complete, calls } = scripted(['{not json', VALID]);
    const res = await generateObject(complete, PROMPT, Answer);
    expect(res).toMatchObject({
      status: 'ok',
      repairs: 1,
      attempts: 2,
      usage: { inputTokens: 20, outputTokens: 10 },
    });
    const second = calls[1]!;
    expect(second.slice(0, 1)).toEqual(PROMPT);
    expect(second[1]).toEqual({ role: 'assistant', content: '{not json' });
    expect(second[2]).toMatchObject({ role: 'user', content: expect.stringContaining('invalid JSON') });
  });

  it('reports invalid_output after 2 repairs (3 calls) when the schema never validates', async () => {
    const wrong = JSON.stringify({ found: 'maybe' });
    const { complete, calls } = scripted([wrong, wrong, wrong, VALID]);
    const res = await generateObject(complete, PROMPT, Answer);
    expect(calls).toHaveLength(3);
    expect(res).toMatchObject({ status: 'invalid_output', repairs: 2, attempts: 3, raw: wrong });
    if (res.status === 'invalid_output') {
      expect(res.error).toContain('found');
    }
  });

  it('treats an empty response as invalid', async () => {
    const { complete } = scripted([]);
    const res = await generateObject(complete, PROMPT, Answer, 0);
    expect(res).toMatchObject({ status: 'invalid_output', error: 'empty response', raw: null });
  });
});
