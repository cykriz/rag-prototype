// Database side of the retrieval eval (docs/PLAN.md M11): checks that the ingest matches the corpus the golden spans
// were validated against, and searches all questions with batched query embeddings.
import type { Sql } from 'postgres';
import type { EmbeddingProvider } from '../llm/types';
import { vectorSearch } from '../retrieval/vector';
import type { RunKpis } from './answers';
import type { RetrievedHit, SearchFn } from './runner';

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

/** Embeds the questions in batches of EMBED_BATCH (one llm_calls row per batch), then one vectorSearch each. */
export function pgVectorSearch(
  sql: Sql,
  embedder: EmbeddingProvider,
  opts: { chunkConfig: string; k: number },
): SearchFn {
  return async (questions) => {
    const results: RetrievedHit[][] = [];
    for (let i = 0; i < questions.length; i += EMBED_BATCH) {
      const batch = questions.slice(i, i + EMBED_BATCH);
      const { vectors } = await embedder.embed(batch, 'query');
      if (vectors.length !== batch.length) {
        throw new Error(
          `Embedding provider returned ${vectors.length} vectors for ${batch.length} questions`,
        );
      }
      for (const vector of vectors) {
        const hits = await vectorSearch(sql, {
          vector,
          k: opts.k,
          chunkConfig: opts.chunkConfig,
          model: embedder.model,
        });
        results.push(
          hits.map((h) => ({
            chunkId: h.chunkId,
            sourcePath: h.sourcePath,
            content: h.content,
            distance: h.distance,
          })),
        );
      }
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
}

/**
 * §8 KPIs of one run for the result file: end-to-end latency percentiles from rag_requests (percentile_cont, as the
 * §8 KPI query) and one llm_calls row per operation and model.
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
           coalesce(sum(cost_usd), 0)::float8                     AS cost_usd
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
  }));
  return { latencyMs: { p50: latency?.p50 ?? null, p95: latency?.p95 ?? null }, calls };
}
