import { APIConnectionTimeoutError, APIError } from 'openai';
import { describe, expect, it } from 'vitest';

import { attemptsOf, isRetryable, withRetry } from '../../src/llm/retry';

const apiError = (status: number) => APIError.generate(status, undefined, `status ${status}`, new Headers());

function failingTimes(n: number, err: () => unknown) {
  let calls = 0;
  const fn = async () => {
    calls++;
    if (calls <= n) {
      throw err();
    }
    return 'ok';
  };
  return { fn, calls: () => calls };
}

function recorder() {
  const delays: number[] = [];
  return { delays, sleep: async (ms: number) => void delays.push(ms), random: () => 1 };
}

describe('withRetry', () => {
  it('retries a 503 twice, then succeeds on attempt 3 with increasing delays', async () => {
    const { fn, calls } = failingTimes(2, () => apiError(503));
    const rec = recorder();
    const res = await withRetry(fn, rec);
    expect(res).toEqual({ value: 'ok', attempts: 3 });
    expect(calls()).toBe(3);
    expect(rec.delays).toEqual([500, 1000]);
  });

  it('applies full jitter between 0 and the exponential cap', async () => {
    const { fn } = failingTimes(1, () => apiError(503));
    const delays: number[] = [];
    await withRetry(fn, { sleep: async (ms) => void delays.push(ms), random: () => 0.25 });
    expect(delays).toEqual([125]);
  });

  it('does not retry a 400', async () => {
    const { fn, calls } = failingTimes(5, () => apiError(400));
    const rec = recorder();
    await expect(withRetry(fn, rec)).rejects.toMatchObject({ status: 400 });
    expect(calls()).toBe(1);
    expect(rec.delays).toEqual([]);
  });

  it('gives up after 4 attempts and rethrows the last error', async () => {
    const { fn, calls } = failingTimes(10, () => apiError(500));
    const rec = recorder();
    await expect(withRetry(fn, rec)).rejects.toMatchObject({ status: 500 });
    expect(calls()).toBe(4);
    expect(rec.delays).toEqual([500, 1000, 2000]);
  });

  it('records the attempt count on the rethrown error', async () => {
    const exhausted = await withRetry(failingTimes(10, () => apiError(500)).fn, recorder()).catch((e) => e);
    expect(attemptsOf(exhausted)).toBe(4);
    const rejected = await withRetry(failingTimes(1, () => apiError(400)).fn, recorder()).catch((e) => e);
    expect(attemptsOf(rejected)).toBe(1);
    expect(attemptsOf(new Error('never retried'))).toBeUndefined();
  });
});

describe('isRetryable', () => {
  it.each([
    ['429', apiError(429)],
    ['500', apiError(500)],
    ['503', apiError(503)],
    ['timeout', new APIConnectionTimeoutError()],
    ['ECONNREFUSED', Object.assign(new Error('refused'), { code: 'ECONNREFUSED' })],
    ['ECONNREFUSED as cause', new TypeError('fetch failed', { cause: { code: 'ECONNREFUSED' } })],
  ])('retries %s', (_name, err) => {
    expect(isRetryable(err)).toBe(true);
  });

  it.each([
    ['400', apiError(400)],
    ['404', apiError(404)],
    ['plain error', new Error('boom')],
  ])('does not retry %s', (_name, err) => {
    expect(isRetryable(err)).toBe(false);
  });
});
