// Exponential backoff with full jitter (docs/PLAN.md M2). The SDK's own retries are off (maxRetries: 0).
import { APIConnectionError } from 'openai';

export interface RetryOptions {
  maxAttempts?: number;
  baseMs?: number;
  factor?: number;
  sleep?: (ms: number) => Promise<void>;
  random?: () => number;
}

const hasCode = (err: unknown, code: string): boolean =>
  typeof err === 'object' && err !== null && 'code' in err && err.code === code;

/** Server not reachable: SDK connection errors (incl. timeouts) or a refused connection. */
export function isConnectionError(err: unknown): boolean {
  return (
    err instanceof APIConnectionError || // includes APIConnectionTimeoutError
    hasCode(err, 'ECONNREFUSED') ||
    (err instanceof Error && hasCode(err.cause, 'ECONNREFUSED'))
  );
}

/** Retry on connection errors and timeouts, 429 and 5xx; never on 4xx such as 400. */
export function isRetryable(err: unknown): boolean {
  if (isConnectionError(err)) {
    return true;
  }
  if (typeof err === 'object' && err !== null && 'status' in err && typeof err.status === 'number') {
    return err.status === 429 || err.status >= 500;
  }
  return false;
}

/** Attempt count of an error rethrown by withRetry; kept beside the error so its type stays unchanged. */
const exhausted = new WeakMap<object, number>();

export function attemptsOf(err: unknown): number | undefined {
  return typeof err === 'object' && err !== null ? exhausted.get(err) : undefined;
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  opts: RetryOptions = {},
): Promise<{ value: T; attempts: number }> {
  const {
    maxAttempts = 4,
    baseMs = 500,
    factor = 2,
    sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
    random = Math.random,
  } = opts;
  for (let attempt = 1; ; attempt++) {
    try {
      return { value: await fn(), attempts: attempt };
    } catch (err) {
      if (attempt >= maxAttempts || !isRetryable(err)) {
        if (typeof err === 'object' && err !== null) {
          exhausted.set(err, attempt);
        }
        throw err;
      }
      await sleep(random() * baseMs * factor ** (attempt - 1));
    }
  }
}
