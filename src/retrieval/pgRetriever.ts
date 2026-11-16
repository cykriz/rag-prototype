// LangChain retriever over searchWith (docs/PLAN.md M8, M14) and multiSearch (M16). The query embeddings go through
// our EmbeddingProvider.
import { Document } from '@langchain/core/documents';
import { BaseRetriever, type BaseRetrieverInput } from '@langchain/core/retrievers';
import type { Sql } from 'postgres';
import type { EmbeddingProvider } from '../llm/types';
import { multiSearch } from './multi';
import { DEFAULT_SEARCH, embedQueries, needsEmbedding, type SearchHit, type SearchMode, searchWith } from './search';

export interface ChunkMetadata {
  chunk_id: number;
  source_path: string;
  heading_path: string;
  /**
   * Search score, higher is better, descending across the returned documents (SearchHit.score); the RRF sum over the
   * query lists for invokeMany (expansion). After a rerank (M17) the documents are in rerank order and keep this
   * score, so it is no longer descending.
   */
  score: number;
}

/** `source_path > heading_path`, or just `source_path` for a chunk without heading. */
export function sourceLabel(meta: Pick<ChunkMetadata, 'source_path' | 'heading_path'>): string {
  return meta.heading_path ? `${meta.source_path} > ${meta.heading_path}` : meta.source_path;
}

export interface PgRetrieverInput extends BaseRetrieverInput {
  sql: Sql;
  embedder: EmbeddingProvider;
  chunkConfig: string;
  k: number;
  /** Default DEFAULT_SEARCH (vector). */
  search?: SearchMode;
}

export class PgRetriever extends BaseRetriever<ChunkMetadata> {
  lc_namespace = ['rag', 'retrievers', 'pg'];

  private readonly sql: Sql;
  private readonly embedder: EmbeddingProvider;
  readonly chunkConfig: string;
  readonly k: number;
  readonly search: SearchMode;

  constructor({ sql, embedder, chunkConfig, k, search = DEFAULT_SEARCH, ...fields }: PgRetrieverInput) {
    super(fields);
    this.sql = sql;
    this.embedder = embedder;
    this.chunkConfig = chunkConfig;
    this.k = k;
    this.search = search;
  }

  async _getRelevantDocuments(query: string): Promise<Document<ChunkMetadata>[]> {
    const vector = (await this.embedQueries([query]))?.[0];
    const hits = await searchWith(this.sql, this.search, {
      query,
      ...(vector && { vector }),
      k: this.k,
      chunkConfig: this.chunkConfig,
      model: this.embedder.model,
    });
    return hits.map(toDocument);
  }

  /**
   * Query expansion (M16): the question and its expansions, embedded in one batched call when the mode needs vectors,
   * searched one by one and fused with RRF (multiSearch); `score` is the RRF sum. Bypasses LangChain callbacks, like
   * every other call site that only uses invoke().
   */
  async invokeMany(queries: readonly string[]): Promise<Document<ChunkMetadata>[]> {
    const vectors = await this.embedQueries(queries);
    const hits = await multiSearch(this.sql, this.search, {
      queries,
      ...(vectors && { vectors }),
      k: this.k,
      chunkConfig: this.chunkConfig,
      model: this.embedder.model,
    });
    return hits.map(toDocument);
  }

  /** One embed call for all queries when the mode needs vectors; otherwise no call and null. */
  private async embedQueries(queries: readonly string[]): Promise<number[][] | null> {
    if (!needsEmbedding(this.search)) {
      return null;
    }
    return embedQueries(this.embedder, queries);
  }
}

const toDocument = (h: SearchHit): Document<ChunkMetadata> =>
  new Document<ChunkMetadata>({
    pageContent: h.content,
    metadata: {
      chunk_id: h.chunkId,
      source_path: h.sourcePath,
      heading_path: h.headingPath,
      score: h.score,
    },
  });
