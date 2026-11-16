// Top k chunks by cosine distance from one model's embeddings table (docs/PLAN.md §4, M8).
import pgvector from 'pgvector';
import type { PendingQuery, Row, Sql, TransactionSql } from 'postgres';
import { tableName } from '../db/embeddingTables';

/** Candidate list size per HNSW scan; at least the largest allowed k. */
export const HNSW_EF_SEARCH = 100;
export const MAX_K = HNSW_EF_SEARCH;

export interface VectorQuery {
  /** Query embedding of `model`, already prefixed by the adapter (`embed([q], 'query')`). */
  vector: number[];
  k: number;
  chunkConfig: string;
  /** Embedding model; selects `emb_<model>`. */
  model: string;
}

export interface VectorHit {
  chunkId: number;
  sourcePath: string;
  headingPath: string;
  content: string;
  distance: number;
}

interface HitRow {
  id: string;
  source_path: string;
  heading_path: string;
  content: string;
  distance: number;
}

/** Shared by vectorSearch and the retrieve CLI. */
export function assertK(k: number, label = 'k'): void {
  if (!Number.isInteger(k) || k < 1 || k > MAX_K) {
    throw new Error(`${label} must be an integer 1-${MAX_K}, got ${k}`);
  }
}

/** Shared by vectorSearch and hybridSearch. */
export function validateVectorQuery(q: VectorQuery): void {
  assertK(q.k);
  if (q.vector.length === 0) {
    throw new Error('Query vector is empty');
  }
}

/**
 * The HNSW scan of the §4 query: `id` and cosine `dist` of the `limit` nearest chunks of `chunkConfig`, in index
 * order (unsorted under relaxed_order). Shared by vectorQuery and the vector side of hybridSearch.
 */
export function vectorCandidates(
  sql: TransactionSql,
  q: Omit<VectorQuery, 'k'>,
  limit: number,
): PendingQuery<Row[]> {
  const vec = pgvector.toSql(q.vector);
  return sql`
    SELECT e.chunk_id AS id, e.embedding <=> ${vec}::vector AS dist
    FROM ${sql(tableName(q.model))} e
    WHERE e.chunk_config = ${q.chunkConfig}
    ORDER BY e.embedding <=> ${vec}::vector
    LIMIT ${limit}`;
}

/**
 * The SELECT of the §4 query as a fragment, run inside the transaction of applyHnswSettings, so tests can EXPLAIN
 * exactly what vectorSearch runs. The table name is safe through `sql()` identifier quoting. The outer sort fixes the
 * order relaxed_order may scramble; `id` breaks ties.
 */
export function vectorQuery(sql: TransactionSql, q: VectorQuery): PendingQuery<Row[]> {
  validateVectorQuery(q);
  return sql`
    SELECT c.id, c.content, c.heading_path, d.source_path, v.dist AS distance
    FROM (${vectorCandidates(sql, q, q.k)}) v
    JOIN chunks c    ON c.id = v.id
    JOIN documents d ON d.id = c.document_id
    ORDER BY distance, c.id`;
}

/**
 * Transaction-local HNSW settings: iterative scan keeps reading the index until k rows pass the chunk_config filter
 * (bounded by hnsw.max_scan_tuples, default 20000), so k rows come back whenever the config holds at least k chunks.
 */
export async function applyHnswSettings(
  tx: TransactionSql,
  iterativeScan: 'relaxed_order' | 'off' = 'relaxed_order',
) {
  await tx.unsafe(`SET LOCAL hnsw.ef_search = ${HNSW_EF_SEARCH}`);
  await tx.unsafe(`SET LOCAL hnsw.iterative_scan = ${iterativeScan}`);
}

export async function vectorSearch(sql: Sql, q: VectorQuery): Promise<VectorHit[]> {
  validateVectorQuery(q);
  const rows = await sql.begin(async (tx) => {
    await applyHnswSettings(tx);
    return (await vectorQuery(tx, q)) as unknown as HitRow[];
  });
  return rows.map((r) => ({
    chunkId: Number(r.id),
    sourcePath: r.source_path,
    headingPath: r.heading_path,
    content: r.content,
    distance: Number(r.distance),
  }));
}

/** Throws the ingest hint when `model` has no vectors for `chunkConfig` (or no table at all). */
export async function requireEmbeddings(sql: Sql, model: string, chunkConfig: string): Promise<void> {
  const table = tableName(model);
  const [reg] = await sql<{ exists: boolean }[]>`SELECT to_regclass(${table}) IS NOT NULL AS exists`;
  let found = false;
  if (reg?.exists) {
    const [row] = await sql<{ found: boolean }[]>`
      SELECT EXISTS (SELECT 1 FROM ${sql(table)} WHERE chunk_config = ${chunkConfig}) AS found`;
    found = row?.found ?? false;
  }
  if (!found) {
    throw new Error(
      `No embeddings for ${chunkConfig} / ${model}: run bun run ingest --chunk ${chunkConfig} --embed ${model}`,
    );
  }
}
