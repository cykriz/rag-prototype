// LangChain retriever over vectorSearch (docs/PLAN.md M8). The query embedding goes through our EmbeddingProvider.
import { Document } from '@langchain/core/documents';
import { BaseRetriever, type BaseRetrieverInput } from '@langchain/core/retrievers';
import type { Sql } from 'postgres';
import type { EmbeddingProvider } from '../llm/types';
import { vectorSearch } from './vector';

export interface ChunkMetadata {
  chunk_id: number;
  source_path: string;
  heading_path: string;
  /** Cosine distance to the query, ascending across the returned documents. */
  distance: number;
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
}

export class PgRetriever extends BaseRetriever<ChunkMetadata> {
  lc_namespace = ['rag', 'retrievers', 'pg'];

  private readonly sql: Sql;
  private readonly embedder: EmbeddingProvider;
  readonly chunkConfig: string;
  readonly k: number;

  constructor({ sql, embedder, chunkConfig, k, ...fields }: PgRetrieverInput) {
    super(fields);
    this.sql = sql;
    this.embedder = embedder;
    this.chunkConfig = chunkConfig;
    this.k = k;
  }

  async _getRelevantDocuments(query: string): Promise<Document<ChunkMetadata>[]> {
    const { vectors } = await this.embedder.embed([query], 'query');
    const [vector] = vectors;
    if (!vector) {
      throw new Error('Embedding provider returned no vector for the query');
    }
    const hits = await vectorSearch(this.sql, {
      vector,
      k: this.k,
      chunkConfig: this.chunkConfig,
      model: this.embedder.model,
    });
    return hits.map(
      (h) =>
        new Document<ChunkMetadata>({
          pageContent: h.content,
          metadata: {
            chunk_id: h.chunkId,
            source_path: h.sourcePath,
            heading_path: h.headingPath,
            distance: h.distance,
          },
        }),
    );
  }
}
