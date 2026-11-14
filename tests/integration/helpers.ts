// Shared seeding for integration tests that need many embedding rows without running the ingest.
import pgvector from 'pgvector';
import type { Sql } from 'postgres';
import { parseChunkConfigKey } from '../../src/ingest/chunk';
import { sha256 } from '../../src/ingest/hash';
import { fakeEmbedding } from '../../src/llm/fake';

/**
 * Inserts one document with `count` synthetic chunks in `config` (creating the chunk_configs row if needed) and their
 * fake nomic embeddings. Chunk text is `${sourcePath} chunk ${i}`, so each call yields distinct vectors.
 */
export async function insertSyntheticChunks(
  sql: Sql,
  opts: { config: string; count: number; sourcePath: string },
): Promise<void> {
  const { chunkSize, chunkOverlap, splitter } = parseChunkConfigKey(opts.config);
  await sql`INSERT INTO chunk_configs (key, chunk_size, chunk_overlap, splitter)
            VALUES (${opts.config}, ${chunkSize}, ${chunkOverlap}, ${splitter})
            ON CONFLICT (key) DO NOTHING`;
  const [doc] = await sql<{ id: string }[]>`
    INSERT INTO documents (source_path, title, content_hash, corpus_version)
    VALUES (${opts.sourcePath}, 'Synthetic', ${sha256(opts.sourcePath)}, 'test') RETURNING id`;
  if (!doc) {
    throw new Error('document insert returned no row');
  }
  const texts = Array.from({ length: opts.count }, (_, i) => `${opts.sourcePath} chunk ${i}`);
  const chunks = await sql<{ id: string; content: string }[]>`
    INSERT INTO chunks ${sql(
      texts.map((content, ordinal) => ({
        document_id: doc.id,
        chunk_config: opts.config,
        ordinal,
        content,
        content_hash: sha256(content),
        token_count: 3,
      })),
    )} RETURNING id, content`;
  await sql`
    INSERT INTO emb_nomic_embed_text ${sql(
      chunks.map((c) => ({
        chunk_id: c.id,
        chunk_config: opts.config,
        content_hash: sha256(c.content),
        embedding: pgvector.toSql(fakeEmbedding(c.content, 768)),
      })),
    )}`;
}
