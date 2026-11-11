import { describe, expect, it } from 'vitest';

import {
  CHAT_MODELS,
  type CheckId,
  type CheckResult,
  EMBED_MODELS,
  exitCode,
  GENERATOR,
  JUDGE,
} from '../../scripts/spike-ollama';

const r = (id: CheckId, model: string, pass = true): CheckResult => ({ id, model, pass, detail: '' });

function allPass(): CheckResult[] {
  const chat = CHAT_MODELS.flatMap((m) => (['a', 'b', 'c', 'f', 'g'] as const).map((id) => r(id, m)));
  const embed = EMBED_MODELS.flatMap(({ model }) => [r('e', model), r('f', model)]);
  return [...chat, r('d', GENERATOR), ...embed];
}

const failing = (results: CheckResult[], id: CheckId, model?: string): CheckResult[] =>
  results.map((x) => (x.id === id && (model === undefined || x.model === model) ? { ...x, pass: false } : x));

describe('spike exitCode (PLAN M1 exit rule)', () => {
  it('returns 0 when every check passes', () => {
    expect(exitCode(allPass())).toBe(0);
  });

  it('returns 0 when the generator passes only via json_object (g)', () => {
    const results = failing(failing(failing(allPass(), 'a', GENERATOR), 'b', GENERATOR), 'c', GENERATOR);
    expect(exitCode(results)).toBe(0);
  });

  it('returns 1 when the judge fails b, c and g', () => {
    const results = failing(failing(failing(allPass(), 'b', JUDGE), 'c', JUDGE), 'g', JUDGE);
    expect(exitCode(results)).toBe(1);
  });

  it('returns 1 when tool calling (d) fails', () => {
    expect(exitCode(failing(allPass(), 'd'))).toBe(1);
  });

  it('returns 1 when one embedding dimension check (e) fails', () => {
    expect(exitCode(failing(allPass(), 'e', 'mxbai-embed-large'))).toBe(1);
  });

  it('returns 1 when usage (f) is missing for one model', () => {
    expect(exitCode(failing(allPass(), 'f', 'nomic-embed-text'))).toBe(1);
  });

  it('treats a missing check as failed', () => {
    expect(exitCode(allPass().filter((x) => x.id !== 'd'))).toBe(1);
  });
});
