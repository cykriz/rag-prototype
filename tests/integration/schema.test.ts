import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import pgvector from 'pgvector';
import type { Sql } from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';

import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { sha256 } from '../../src/ingest/hash';
import { fakeEmbedding } from '../../src/llm/fake';

let sql: Sql;

/** Everything a rerun could change: columns, indexes and recorded migrations. */
async function schemaSnapshot(): Promise<unknown> {
  const columns = await sql`
    SELECT table_name, column_name, data_type, udt_name, is_nullable, column_default
    FROM information_schema.columns WHERE table_schema = 'public'
    ORDER BY table_name, column_name`;
  const indexes = await sql`
    SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' ORDER BY indexname`;
  const versions = await sql`SELECT version, applied_at FROM schema_migrations ORDER BY version`;
  return { columns: [...columns], indexes: [...indexes], versions: [...versions] };
}

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  // Fresh schema per run, so the first migrate below really starts from nothing.
  await sql.unsafe('DROP SCHEMA public CASCADE; CREATE SCHEMA public');
});

afterAll(async () => {
  await sql?.end();
});

describe('migrations and embedding tables', () => {
  it('apply once; a second run changes nothing', async () => {
    expect(await migrate(sql)).toEqual(['001_init']);
    expect(await ensureEmbeddingTables(sql)).toEqual(['emb_nomic_embed_text', 'emb_mxbai_embed_large']);
    const before = await schemaSnapshot();

    expect(await migrate(sql)).toEqual([]);
    await ensureEmbeddingTables(sql);
    expect(await schemaSnapshot()).toEqual(before);
  });

  it('installs pgvector 0.8 or newer', async () => {
    const [row] = await sql<{ extversion: string }[]>`
      SELECT extversion FROM pg_extension WHERE extname = 'vector'`;
    const [major = 0, minor = 0] = (row?.extversion ?? '0').split('.').map(Number);
    expect(major > 0 || minor >= 8).toBe(true);
  });

  it('rejects a 1024-float vector in the 768-dim nomic table', async () => {
    const vector = pgvector.toSql(fakeEmbedding('wrong model', 1024));
    await expect(
      sql`INSERT INTO emb_nomic_embed_text (chunk_id, chunk_config, content_hash, embedding)
          VALUES (1, 'md-512-64', ${sha256('x')}, ${vector}::vector)`,
    ).rejects.toThrow(/expected 768 dimensions/);
  });

  it('aborts when an existing table has a different dim than the config', async () => {
    await expect(ensureEmbeddingTables(sql, { 'nomic-embed-text': { dim: 1024 } })).rejects.toThrow(
      /emb_nomic_embed_text\.embedding has dim 768, config\/models\.json says 1024/,
    );
  });

  it('surfaces the SQL error of a broken migration and records nothing', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'migrate-'));
    try {
      await writeFile(
        join(dir, '900_broken.sql'),
        `BEGIN;
         CREATE TABLE broken_probe (id int);
         SELECT * FROM missing_relation;
         INSERT INTO schema_migrations (version) VALUES ('900_broken');
         COMMIT;`,
      );
      await expect(migrate(sql, dir)).rejects.toThrow(/900_broken\.sql failed: .*missing_relation/);
      const [row] = await sql<{ probe: string | null; recorded: boolean }[]>`
        SELECT to_regclass('broken_probe')::text AS probe,
               EXISTS (SELECT 1 FROM schema_migrations WHERE version = '900_broken') AS recorded`;
      expect(row).toEqual({ probe: null, recorded: false });
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});

describe('HNSW index', () => {
  const CHUNK_CONFIG = 'md-512-64';
  const ROWS = 2000;

  beforeAll(async () => {
    await sql`INSERT INTO chunk_configs (key, chunk_size, chunk_overlap, splitter)
              VALUES (${CHUNK_CONFIG}, 512, 64, 'markdown')`;
    const [doc] = await sql<{ id: string }[]>`
      INSERT INTO documents (source_path, title, content_hash, corpus_version)
      VALUES ('docs/index-check.md', 'Index check', ${sha256('doc')}, 'test') RETURNING id`;
    const texts = Array.from({ length: ROWS }, (_, i) => `synthetic chunk ${i}`);
    const chunks = await sql<{ id: string; content: string }[]>`
      INSERT INTO chunks ${sql(
        texts.map((content, ordinal) => ({
          document_id: doc!.id,
          chunk_config: CHUNK_CONFIG,
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
          chunk_config: CHUNK_CONFIG,
          content_hash: sha256(c.content),
          embedding: pgvector.toSql(fakeEmbedding(c.content, 768)),
        })),
      )}`;
    await sql`ANALYZE emb_nomic_embed_text, chunks`;
  });

  it('serves the filtered vector query', async () => {
    const literal = `'${pgvector.toSql(fakeEmbedding('how do I read env vars', 768))}'::vector`;
    // Statements and SELECT copied from docs/PLAN.md §4. Once src/retrieval/vector.ts exists (M8), it exports
    // them and this test imports them instead.
    const plan = await sql.begin(async (tx) => {
      await tx.unsafe('SET LOCAL hnsw.ef_search = 100');
      await tx.unsafe('SET LOCAL hnsw.iterative_scan = relaxed_order');
      return tx.unsafe(`EXPLAIN ANALYZE
        SELECT * FROM (
          SELECT c.id, c.content, c.heading_path, d.source_path,
                 e.embedding <=> ${literal} AS distance
          FROM emb_nomic_embed_text e
          JOIN chunks c    ON c.id = e.chunk_id
          JOIN documents d ON d.id = c.document_id
          WHERE e.chunk_config = '${CHUNK_CONFIG}'
          ORDER BY e.embedding <=> ${literal}
          LIMIT 5
        ) t ORDER BY distance`);
    });
    const text = plan.map((r) => r['QUERY PLAN']).join('\n');
    expect(text).toContain('emb_nomic_embed_text_hnsw');
  });
});
