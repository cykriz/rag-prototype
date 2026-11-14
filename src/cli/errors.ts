// Error messages shared by the CLIs that call Ollama (ask:raw, ingest).
import { APIError } from 'openai';
import { isConnectionError } from '../llm/retry';

/** `model` is the one the failed call used, named in the 404 hint (chat model in ask:raw, embedding model in ingest). */
export function describeError(err: unknown, ctx: { baseURL: string; model: string }): string {
  if (isConnectionError(err)) {
    return `Ollama not reachable at ${ctx.baseURL}, start it (ollama serve)`;
  }
  if (err instanceof APIError) {
    const hint = err.status === 404 ? ` (is the model pulled? ollama pull ${ctx.model})` : '';
    return `Model call failed with status ${err.status}: ${err.message}${hint}`;
  }
  return err instanceof Error ? err.message : String(err);
}
