// Hybrid search (docs/PLAN.md M14): reciprocal rank fusion of the vector and the full text ranking, in SQL.
import type { Sql } from 'postgres';
import { ftsCandidates } from './fts';
import { RRF_K } from './rrf';
import { applyHnswSettings, HNSW_EF_SEARCH, validateVectorQuery, vectorCandidates } from './vector';

/** Candidates per side before fusion; raised to k so k results are always reachable. */
const HYBRID_MIN_DEPTH = 50;
// ef_search bounds what one HNSW scan returns; assertK caps k at MAX_K = HNSW_EF_SEARCH, so only this can break it.
if (HYBRID_MIN_DEPTH > HNSW_EF_SEARCH) {
  throw new Error(`HYBRID_MIN_DEPTH ${HYBRID_MIN_DEPTH} exceeds hnsw.ef_search ${HNSW_EF_SEARCH}`);
}

/** max(k, 50): never above ef_search, so the HNSW scan can deliver every vector candidate. */
export const hybridDepth = (k: number): number => Math.max(k, HYBRID_MIN_DEPTH);

export interface HybridQuery {
  /** Query embedding of `model`, already prefixed by the adapter (`embed([q], 'query')`). */
  vector: number[];
  /** Raw query text for websearch_to_tsquery. */
  query: string;
  k: number;
  chunkConfig: string;
  model: string;
}

export interface HybridHit {
  chunkId: number;
  sourcePath: string;
  headingPath: string;
  content: string;
  /** RRF score Σ 1/(RRF_K + rank), higher is better. */
  score: number;
}

interface HybridRow {
  id: string;
  source_path: string;
  heading_path: string;
  content: string;
  score: number;
}

/**
 * Top `hybridDepth(k)` per side, ranked from 1 (vector by distance, FTS by ts_rank_cd, ties by id), fused with
 * sum(1/(RRF_K + r)). An empty FTS side (stop words only, no match) leaves the vector order unchanged. Same tie
 * order as rrf(): score descending, then id.
 */
export async function hybridSearch(sql: Sql, q: HybridQuery): Promise<HybridHit[]> {
  validateVectorQuery(q);
  const depth = hybridDepth(q.k);
  const rows = await sql.begin(async (tx) => {
    await applyHnswSettings(tx);
    return (await tx`
      WITH v AS (
        SELECT id, row_number() OVER (ORDER BY dist, id) AS r FROM (${vectorCandidates(tx, q, depth)}) s
      ), f AS (
        SELECT id, row_number() OVER (ORDER BY rank DESC, id) AS r FROM (${ftsCandidates(tx, q, depth)}) s
      ), fused AS (
        SELECT id, sum(1.0::float8 / (${RRF_K}::int + r)) AS score
        FROM (SELECT * FROM v UNION ALL SELECT * FROM f) u
        GROUP BY id
      )
      SELECT c.id, c.content, c.heading_path, d.source_path, fused.score
      FROM fused
      JOIN chunks c    ON c.id = fused.id
      JOIN documents d ON d.id = c.document_id
      ORDER BY fused.score DESC, c.id
      LIMIT ${q.k}`) as unknown as HybridRow[];
  });
  return rows.map((r) => ({
    chunkId: Number(r.id),
    sourcePath: r.source_path,
    headingPath: r.heading_path,
    content: r.content,
    score: Number(r.score),
  }));
}
