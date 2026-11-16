// Multi-query search (docs/PLAN.md M16): one searchWith per query, each an SQL search of its own, fused with
// reciprocal rank fusion in TypeScript (rrf()). Shared by PgRetriever.invokeMany and the retrieval eval.
import type { Sql } from 'postgres';
import { hybridDepth } from './hybrid';
import { rrf } from './rrf';
import { type SearchHit, type SearchMode, type SearchQuery, searchWith } from './search';

/**
 * RRF over ranked hit lists (best first): a chunk found by several lists outranks one found once at a similar rank.
 * Order and ties as rrf() (score descending, then chunk id); cut to k. Each fused hit keeps the content and source of
 * its first occurrence and carries the RRF sum as `score`.
 */
export function fuseHits(lists: readonly (readonly SearchHit[])[], k: number): SearchHit[] {
  const byId = new Map<number, SearchHit>();
  for (const list of lists) {
    for (const h of list) {
      if (!byId.has(h.chunkId)) {
        byId.set(h.chunkId, h);
      }
    }
  }
  return rrf(
    lists.map((l) => l.map((h) => h.chunkId)),
    k,
  ).map(({ chunkId, score }) => {
    const hit = byId.get(chunkId);
    if (!hit) {
      throw new Error(`fuseHits: chunk ${chunkId} has no hit`);
    }
    return { ...hit, score };
  });
}

export interface MultiSearchQuery extends Omit<SearchQuery, 'query' | 'vector'> {
  /** The question first, then its expansions; at least one. */
  queries: readonly string[];
  /** One embedding per query, same order; required when needsEmbedding(mode). */
  vectors?: readonly number[][];
}

/**
 * Searches each query in `mode` to depth max(k, 50) (hybridDepth, as the sides of hybrid search) and fuses the lists
 * with fuseHits, cut to k. One query still goes through the fusion, so the scores of an expand run are RRF sums
 * throughout, fallbacks included.
 */
export async function multiSearch(sql: Sql, mode: SearchMode, q: MultiSearchQuery): Promise<SearchHit[]> {
  if (q.queries.length === 0) {
    throw new Error('multiSearch needs at least one query');
  }
  if (q.vectors && q.vectors.length !== q.queries.length) {
    throw new Error(`multiSearch got ${q.vectors.length} vectors for ${q.queries.length} queries`);
  }
  const depth = hybridDepth(q.k);
  const lists: SearchHit[][] = [];
  for (const [i, query] of q.queries.entries()) {
    const vector = q.vectors?.[i];
    lists.push(
      await searchWith(sql, mode, {
        query,
        ...(vector && { vector }),
        k: depth,
        chunkConfig: q.chunkConfig,
        model: q.model,
      }),
    );
  }
  return fuseHits(lists, q.k);
}
