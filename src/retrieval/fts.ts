// Full text search over chunks.tsv (docs/PLAN.md M14): websearch_to_tsquery on the english config, ranked by
// ts_rank_cd. Uses the generated tsv column and its GIN index from migrations/001_init.sql.
import type { PendingQuery, Row, Sql, TransactionSql } from 'postgres';
import { assertK } from './vector';

/** Must equal the config of the generated `chunks.tsv` column in migrations/001_init.sql. */
export const TS_CONFIG = 'english';

export interface FtsQuery {
  /** Raw user text; websearch_to_tsquery accepts any input (quotes, OR, -term) without a syntax error. */
  query: string;
  k: number;
  chunkConfig: string;
}

export interface FtsHit {
  chunkId: number;
  sourcePath: string;
  headingPath: string;
  content: string;
  /** ts_rank_cd, higher is better. */
  rank: number;
}

interface FtsRow {
  id: string;
  source_path: string;
  heading_path: string;
  content: string;
  rank: number;
}

/**
 * `id` and `rank` (ts_rank_cd as float8) of the top `limit` matching chunks of `chunkConfig`, ordered rank
 * descending, then id. Shared by ftsSearch and the FTS side of hybridSearch.
 */
export function ftsCandidates(
  sql: Sql | TransactionSql,
  q: Omit<FtsQuery, 'k'>,
  limit: number,
): PendingQuery<Row[]> {
  return sql`
    SELECT c.id, ts_rank_cd(c.tsv, q)::float8 AS rank
    FROM chunks c, websearch_to_tsquery(${TS_CONFIG}::regconfig, ${q.query}) q
    WHERE c.chunk_config = ${q.chunkConfig} AND c.tsv @@ q
    ORDER BY rank DESC, c.id
    LIMIT ${limit}`;
}

/**
 * Top k chunks matching the query. A query of stop words only yields an empty tsquery that matches nothing, so the
 * result is empty; fewer than k matches give fewer than k hits. Ties are broken by chunk id.
 */
export async function ftsSearch(sql: Sql, q: FtsQuery): Promise<FtsHit[]> {
  assertK(q.k);
  const rows = (await sql`
    SELECT c.id, c.content, c.heading_path, d.source_path, f.rank
    FROM (${ftsCandidates(sql, q, q.k)}) f
    JOIN chunks c    ON c.id = f.id
    JOIN documents d ON d.id = c.document_id
    ORDER BY f.rank DESC, c.id`) as unknown as FtsRow[];
  return rows.map((r) => ({
    chunkId: Number(r.id),
    sourcePath: r.source_path,
    headingPath: r.heading_path,
    content: r.content,
    rank: Number(r.rank),
  }));
}
