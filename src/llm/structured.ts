// Structured output validation and repair (docs/PLAN.md M1/M2). The request side (zodResponseFormat, path (b)
// in docs/SPIKE.md) lives in the adapter; validation lives here so every provider shares it.
import type { z } from 'zod';
import type { Message, ObjectResult, Usage } from './types';

export interface RawCompletion {
  content: string | null;
  model: string;
  attempts: number;
  usage?: Usage;
}

export function validate<T>(
  raw: string | null,
  schema: z.ZodType<T>,
): { ok: true; value: T } | { ok: false; error: string } {
  if (!raw) {
    return { ok: false, error: 'empty response' };
  }
  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch (err) {
    return { ok: false, error: `invalid JSON: ${err instanceof Error ? err.message : String(err)}` };
  }
  const parsed = schema.safeParse(json);
  return parsed.success ? { ok: true, value: parsed.data } : { ok: false, error: parsed.error.message };
}

const addUsage = (a: Usage | undefined, b: Usage | undefined): Usage | undefined =>
  a && b
    ? { inputTokens: a.inputTokens + b.inputTokens, outputTokens: a.outputTokens + b.outputTokens }
    : (a ?? b);

/**
 * Calls `complete` and validates the JSON against `schema`. On failure, appends the invalid output and the
 * error to the conversation and asks again, up to `maxRepairs` times; then reports `invalid_output`.
 */
export async function generateObject<T>(
  complete: (messages: Message[]) => Promise<RawCompletion>,
  messages: Message[],
  schema: z.ZodType<T>,
  maxRepairs = 2,
): Promise<ObjectResult<T>> {
  let convo = messages;
  let attempts = 0;
  let usage: Usage | undefined;
  for (let repairs = 0; ; repairs++) {
    const res = await complete(convo);
    attempts += res.attempts;
    usage = addUsage(usage, res.usage);
    const v = validate(res.content, schema);
    const info = { model: res.model, attempts, repairs, ...(usage && { usage }) };
    if (v.ok) {
      return { ...info, status: 'ok', value: v.value };
    }
    if (repairs >= maxRepairs) {
      return { ...info, status: 'invalid_output', error: v.error, raw: res.content };
    }
    convo = [
      ...convo,
      // Ollama rejects assistant content null without tool_calls (400), so send '' instead.
      { role: 'assistant', content: res.content ?? '' },
      {
        role: 'user',
        content: `Your previous output was invalid: ${v.error}\nReply again with only JSON that matches the schema.`,
      },
    ];
  }
}
