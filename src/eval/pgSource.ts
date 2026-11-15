// Database side of the retrieval eval (docs/PLAN.md M11): checks that the ingest matches the corpus the golden spans
// were validated against, and searches all questions with batched query embeddings.
import type { Sql } from 'postgres';
import type { EmbeddingProvider } from '../llm/types';
import { vectorSearch } from '../retrieval/vector';
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
