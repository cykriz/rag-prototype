import { join } from 'node:path';
import type { Sql } from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';

import { loadConfig } from '../../src/config';
import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { parseChunkConfigKey } from '../../src/ingest/chunk';
import { embedText } from '../../src/ingest/hash';
import { ingest } from '../../src/ingest/ingest';
import { FakeEmbeddingProvider, fakeEmbedding } from '../../src/llm/fake';
import { PgRetriever } from '../../src/retrieval/pgRetriever';
import {
  applyHnswSettings,
  HNSW_EF_SEARCH,
  requireEmbeddings,
  type VectorQuery,
  vectorQuery,
  vectorSearch,
} from '../../src/retrieval/vector';
import { insertSyntheticChunks } from './helpers';

const FIXTURES = join(import.meta.dirname, '../fixtures/docs');
const NOMIC = loadConfig({ EMBED_PROVIDER: 'fake' });
const FIXTURE_CONFIG = parseChunkConfigKey('md-64-0');
/** Two synthetic configs of equal size: the filter drops half of every HNSW candidate list. */
const TARGET = 'md-256-32';
const OTHER = 'md-1024-128';
const SYNTHETIC_ROWS = 1000;
const MODEL = 'nomic-embed-text';

let sql: Sql;
const embedder = new FakeEmbeddingProvider(MODEL, NOMIC.embedding.dim);

const isAscending = (xs: number[]) => xs.every((x, i) => i === 0 || (xs[i - 1] ?? x) <= x);

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  await migrate(sql);
  await ensureEmbeddingTables(sql);
  await sql`TRUNCATE documents, chunk_configs CASCADE`;
  await ingest({
    sql,
    root: FIXTURES,
    chunkConfig: FIXTURE_CONFIG,
    embedder,
    embedding: NOMIC.embedding,
    corpusVersion: 'fixture-v1',
    warn: () => {},
  });
  await insertSyntheticChunks(sql, {
    config: TARGET,
    count: SYNTHETIC_ROWS,
    sourcePath: 'synthetic/target.md',
  });
  await insertSyntheticChunks(sql, {
    config: OTHER,
    count: SYNTHETIC_ROWS,
    sourcePath: 'synthetic/other.md',
  });
  await sql`ANALYZE emb_nomic_embed_text, chunks, documents`;
});

afterAll(async () => {
  await sql?.end();
});

describe('PgRetriever', () => {
  it('ranks the chunk whose embed text equals the query first', async () => {
    const [chunk] = await sql<{ id: string; heading_path: string; content: string; source_path: string }[]>`
      SELECT c.id, c.heading_path, c.content, d.source_path
      FROM chunks c JOIN documents d ON d.id = c.document_id
      WHERE c.chunk_config = ${FIXTURE_CONFIG.key} AND c.heading_path <> ''
      ORDER BY c.id LIMIT 1`;
    if (!chunk) {
      throw new Error('fixture corpus has no chunk with a heading path');
    }
    const retriever = new PgRetriever({ sql, embedder, chunkConfig: FIXTURE_CONFIG.key, k: 5 });

    const docs = await retriever.invoke(
      embedText({ headingPath: chunk.heading_path, content: chunk.content }),
    );

    expect(docs).toHaveLength(5);
    expect(docs[0]?.pageContent).toBe(chunk.content);
    expect(docs[0]?.metadata).toEqual({
      chunk_id: Number(chunk.id),
      source_path: chunk.source_path,
      heading_path: chunk.heading_path,
      score: expect.closeTo(1, 6),
    });
    // Vector score = 1 − cosine distance, so it descends.
    expect(isAscending(docs.map((d) => -d.metadata.score))).toBe(true);

    // The content alone is not the embed text: the fake vector differs, so the score is below 1.
    const contentOnly = await retriever.invoke(chunk.content);
    expect(contentOnly[0]?.metadata.score).toBeLessThan(1 - 1e-6);
  });
});

describe('vectorSearch', () => {
  const query = (k: number, chunkConfig = TARGET): VectorQuery => ({
    vector: fakeEmbedding('how do I read env vars', NOMIC.embedding.dim),
    k,
    chunkConfig,
    model: MODEL,
  });

  it('returns k rows of the filtered config, whatever plan the planner picks', async () => {
    const hits = await vectorSearch(sql, query(HNSW_EF_SEARCH));

    expect(hits).toHaveLength(HNSW_EF_SEARCH);
    const configs = await sql<{ chunk_config: string }[]>`
      SELECT DISTINCT chunk_config FROM chunks WHERE id IN ${sql(hits.map((h) => h.chunkId))}`;
    expect(configs.map((r) => r.chunk_config)).toEqual([TARGET]);
    expect(isAscending(hits.map((h) => h.distance))).toBe(true);
  });

  it('on the HNSW path, iterative scan is what keeps k rows when the filter drops candidates', async () => {
    // At k = 100 on 2000 rows the planner rightly prefers an exact sort, which also returns k rows. To exercise the
    // index path, use a small k (planner picks HNSW) and shrink ef_search to k: one candidate list then holds k rows,
    // and the 50 % filter drops about half of them unless iterative scan keeps reading.
    const k = 10;
    const run = (iterativeScan: 'relaxed_order' | 'off', explain: boolean) =>
      sql.begin(async (tx) => {
        await applyHnswSettings(tx, iterativeScan);
        await tx.unsafe(`SET LOCAL hnsw.ef_search = ${k}`);
        const q = vectorQuery(tx, query(k));
        return explain ? tx`EXPLAIN ${q}` : tx`${q}`;
      });

    const plan = await run('relaxed_order', true);
    expect(plan.map((r) => r['QUERY PLAN']).join('\n')).toContain('emb_nomic_embed_text_hnsw');
    const rows = await run('relaxed_order', false);
    expect(rows).toHaveLength(k);
    expect(isAscending(rows.map((r) => Number(r.distance)))).toBe(true);

    // Control: one candidate list, filtered once, loses rows.
    expect((await run('off', false)).length).toBeLessThan(k);
  });

  it('returns every row of a config smaller than k', async () => {
    const [row] = await sql<{ n: number }[]>`
      SELECT count(*)::int AS n FROM emb_nomic_embed_text WHERE chunk_config = ${FIXTURE_CONFIG.key}`;
    const total = row?.n ?? 0;
    expect(total).toBeGreaterThan(0);
    expect(total).toBeLessThan(HNSW_EF_SEARCH);

    const hits = await vectorSearch(sql, query(HNSW_EF_SEARCH, FIXTURE_CONFIG.key));

    expect(hits).toHaveLength(total);
    expect(isAscending(hits.map((h) => h.distance))).toBe(true);
  });

  it('rejects k outside 1 to ef_search', async () => {
    await expect(vectorSearch(sql, query(0))).rejects.toThrow(/k must be an integer 1-100, got 0/);
    await expect(vectorSearch(sql, query(HNSW_EF_SEARCH + 1))).rejects.toThrow(/got 101/);
  });
});

describe('requireEmbeddings', () => {
  it('passes for an ingested config and model', async () => {
    await expect(requireEmbeddings(sql, MODEL, FIXTURE_CONFIG.key)).resolves.toBeUndefined();
  });

  it('names the ingest command for a config or model without vectors', async () => {
    await expect(requireEmbeddings(sql, MODEL, 'md-512-64')).rejects.toThrow(
      'No embeddings for md-512-64 / nomic-embed-text: run bun run ingest --chunk md-512-64 --embed nomic-embed-text',
    );
    await expect(requireEmbeddings(sql, 'mxbai-embed-large', FIXTURE_CONFIG.key)).rejects.toThrow(
      `No embeddings for md-64-0 / mxbai-embed-large: run bun run ingest --chunk md-64-0 --embed mxbai-embed-large`,
    );
  });
});
