// One entry point for the three search modes (docs/PLAN.md M14), so the eval, the retriever and the CLIs share the
// dispatch and the hit shape.
import type { Sql } from 'postgres';
import { ftsSearch } from './fts';
import { hybridSearch } from './hybrid';
import { vectorSearch } from './vector';

export const SEARCH_MODES = ['vector', 'fts', 'hybrid'] as const;
export type SearchMode = (typeof SEARCH_MODES)[number];
export const DEFAULT_SEARCH: SearchMode = 'vector';

export interface SearchHit {
  chunkId: number;
  sourcePath: string;
  headingPath: string;
  content: string;
  /**
   * Higher is better in every mode, comparable only within one mode: vector 1 − cosine distance, fts ts_rank_cd,
   * hybrid the RRF sum.
   */
  score: number;
}

export interface SearchQuery {
  query: string;
  /** Query embedding of `model`; required when needsEmbedding(mode). */
  vector?: number[];
  k: number;
  chunkConfig: string;
  model: string;
}

/** fts runs on the query text alone and makes no embedding call. */
export const needsEmbedding = (mode: SearchMode): boolean => mode !== 'fts';

function requireVector(mode: SearchMode, q: SearchQuery): number[] {
  if (!q.vector) {
    throw new Error(`search mode ${mode} needs a query vector`);
  }
  return q.vector;
}

export async function searchWith(sql: Sql, mode: SearchMode, q: SearchQuery): Promise<SearchHit[]> {
  switch (mode) {
    case 'vector': {
      const hits = await vectorSearch(sql, {
        vector: requireVector(mode, q),
        k: q.k,
        chunkConfig: q.chunkConfig,
        model: q.model,
      });
      return hits.map(({ distance, ...h }) => ({ ...h, score: 1 - distance }));
    }
    case 'fts': {
      const hits = await ftsSearch(sql, { query: q.query, k: q.k, chunkConfig: q.chunkConfig });
      return hits.map(({ rank, ...h }) => ({ ...h, score: rank }));
    }
    case 'hybrid':
      return hybridSearch(sql, {
        vector: requireVector(mode, q),
        query: q.query,
        k: q.k,
        chunkConfig: q.chunkConfig,
        model: q.model,
      });
  }
}
