// Database side of the retrieval eval (docs/PLAN.md M11): checks that the ingest matches the corpus the golden spans
// were validated against, and searches all questions (batched query embeddings where the search mode needs them).
import type { Sql } from 'postgres';
import type { EmbeddingProvider } from '../llm/types';
import { multiSearch } from '../retrieval/multi';
import { embedQueries, needsEmbedding, type SearchHit, type SearchMode, searchWith } from '../retrieval/search';
import type { RunKpis } from './answers';
import type { SearchFn } from './runner';

/** Same batch size as ingest. */
export const EMBED_BATCH = 32;

/**
 * Throws when chunks of `chunkConfig` come from another corpus version than `corpusSha`, or when an evidence
 * source has no chunks in `chunkConfig`: the labels would not fit the ingest. Call after requireEmbeddings.
 */
export async function checkIngest(
  sql: Sql,
  opts: { chunkConfig: string; corpusSha: string; sources: readonly string[] },
): Promise<void> {
  const versions = await sql<{ corpus_version: string }[]>`
    SELECT DISTINCT d.corpus_version FROM documents d
    WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id AND c.chunk_config = ${opts.chunkConfig})
    ORDER BY 1`;
  const other = versions.map((v) => v.corpus_version).filter((v) => v !== opts.corpusSha);
  if (other.length > 0) {
    throw new Error(
      `Chunks of ${opts.chunkConfig} come from corpus ${other.join(', ')}, CORPUS_SHA is ${opts.corpusSha}: ` +
        `re-run bun run ingest --chunk ${opts.chunkConfig}`,
    );
  }
  const wanted = [...new Set(opts.sources)].sort();
  const present = await sql<{ source_path: string }[]>`
    SELECT d.source_path FROM documents d
    WHERE d.source_path = ANY(${wanted})
      AND EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id AND c.chunk_config = ${opts.chunkConfig})`;
  const have = new Set(present.map((r) => r.source_path));
  const missing = wanted.filter((s) => !have.has(s));
  if (missing.length > 0) {
    throw new Error(
      `No chunks in ${opts.chunkConfig} for evidence sources ${missing.join(', ')}: ` +
        `re-run bun run ingest --chunk ${opts.chunkConfig}`,
    );
  }
}

/**
 * One search per question in `mode`, to the depth the runner asks for (retrievalDepth: k, or the rerank candidates,
 * M17). Modes with a query vector embed all queries of all questions in batches of EMBED_BATCH first (one llm_calls
 * row per batch); fts makes no embedding call. Without `fuse` every question has exactly one query and is searched
 * once, as before M16; with `fuse` (expand on) its queries go through multiSearch, fused with RRF even when only the
 * question is left.
 */
export function pgSearch(
  sql: Sql,
  embedder: EmbeddingProvider,
  opts: { mode: SearchMode; chunkConfig: string; fuse?: boolean },
): SearchFn {
  return async (queryLists, depth) => {
    if (!opts.fuse && queryLists.some((qs) => qs.length !== 1)) {
      throw new Error('pgSearch without fuse takes exactly one query per question');
    }
    const flat = queryLists.flat();
    let vectors: number[][] | null = null;
    if (needsEmbedding(opts.mode)) {
      vectors = [];
      for (let i = 0; i < flat.length; i += EMBED_BATCH) {
        vectors.push(...(await embedQueries(embedder, flat.slice(i, i + EMBED_BATCH))));
      }
    }
    const results: SearchHit[][] = [];
    let offset = 0;
    for (const queries of queryLists) {
      const own = vectors?.slice(offset, offset + queries.length);
      offset += queries.length;
      let hits: SearchHit[];
      if (opts.fuse) {
        hits = await multiSearch(sql, opts.mode, {
          queries,
          ...(own && { vectors: own }),
          k: depth,
          chunkConfig: opts.chunkConfig,
          model: embedder.model,
        });
      } else {
        const [query] = queries;
        if (query === undefined) {
          throw new Error('pgSearch got a question without a query');
        }
        const vector = own?.[0];
        hits = await searchWith(sql, opts.mode, {
          query,
          ...(vector && { vector }),
          k: depth,
          chunkConfig: opts.chunkConfig,
          model: embedder.model,
        });
      }
      results.push(hits);
    }
    return results;
  };
}

interface KpiRow {
  operation: string;
  model: string;
  calls: number;
  avg_in: number | null;
  avg_out: number | null;
  estimated: number;
  retry_rate: number;
  invalid_rate: number;
  cost_usd: number;
  latency_p50: number | null;
  latency_p95: number | null;
}

/**
 * §8 KPIs of one run for the result file: end-to-end latency percentiles from rag_requests (percentile_cont, as the
 * §8 KPI query) and one llm_calls row per operation and model, with its call latency percentiles.
 */
export async function fetchRunKpis(sql: Sql, runId: string): Promise<RunKpis> {
  const [latency] = await sql<{ p50: number | null; p95: number | null }[]>`
    SELECT percentile_cont(0.5)  WITHIN GROUP (ORDER BY total_latency_ms) AS p50,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms) AS p95
    FROM rag_requests WHERE run_id = ${runId}`;
  const rows = await sql<KpiRow[]>`
    SELECT operation, model,
           count(*)::int                                          AS calls,
           avg(input_tokens)::float8                              AS avg_in,
           avg(output_tokens)::float8                             AS avg_out,
           avg((token_source = 'tiktoken_estimate')::int)::float8 AS estimated,
           avg((attempts > 1)::int)::float8                       AS retry_rate,
           avg((status = 'invalid_output')::int)::float8          AS invalid_rate,
           coalesce(sum(cost_usd), 0)::float8                     AS cost_usd,
           percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS latency_p50,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS latency_p95
    FROM llm_calls WHERE run_id = ${runId}
    GROUP BY operation, model ORDER BY operation, model`;
  const calls = rows.map((r) => ({
    operation: r.operation,
    model: r.model,
    calls: r.calls,
    avgInputTokens: r.avg_in,
    avgOutputTokens: r.avg_out,
    estimatedTokenShare: r.estimated,
    retryRate: r.retry_rate,
    invalidOutputRate: r.invalid_rate,
    costUsd: r.cost_usd,
    latencyP50Ms: r.latency_p50,
    latencyP95Ms: r.latency_p95,
  }));
  return { latencyMs: { p50: latency?.p50 ?? null, p95: latency?.p95 ?? null }, calls };
}
