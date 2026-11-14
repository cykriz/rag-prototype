// Error messages shared by the CLIs: Ollama errors (ask:raw, ingest, retrieve, ask) and the database probe
// (retrieve, ask).
import { APIError } from 'openai';
import type { Sql } from 'postgres';
import { isConnectionError } from '../llm/retry';

export const DB_DOWN = 'Database not reachable at DATABASE_URL: start it with `docker compose up -d`';

export async function databaseReachable(sql: Sql): Promise<boolean> {
  try {
    await sql`SELECT 1`;
    return true;
  } catch {
    return false;
  }
}

/** isConnectionError matches a refused Postgres connection too; probe the database before blaming Ollama. */
export async function isDatabaseDown(err: unknown, sql: Sql): Promise<boolean> {
  return isConnectionError(err) && !(await databaseReachable(sql));
}

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
