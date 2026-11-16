// Error messages shared by the CLIs: Ollama errors (ask:raw, ingest, retrieve, ask) and the database probe
// (retrieve, ask, eval:*).
import { APIError } from 'openai';
import type { Sql } from 'postgres';
import type { Config } from '../config';
import { isConnectionError } from '../llm/retry';
import { RetrievalError } from '../rag/answer';
import { ChatStageError } from '../rag/stage';

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

/** The provider error inside a RetrievalError or ChatStageError, which the connection and status probes check. */
export const unwrapStageError = (err: unknown): unknown =>
  err instanceof RetrievalError || err instanceof ChatStageError ? err.cause : err;

/**
 * Message for an error of a retrieval-only run (retrieve, eval:retrieval): a ChatStageError (rewrite, expand) is a
 * chat-model failure, anything else is named against the embedding model. Probes the database with the unwrapped error.
 */
export async function retrievalErrorMessage(
  err: unknown,
  config: Pick<Config, 'baseURL' | 'chatModel' | 'embedModel'>,
  sql: Sql,
): Promise<string> {
  const inner = unwrapStageError(err);
  if (await isDatabaseDown(inner, sql)) {
    return DB_DOWN;
  }
  const model = err instanceof ChatStageError ? config.chatModel : config.embedModel;
  return describeError(inner, { baseURL: config.baseURL, model });
}
