import type { Sql } from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';

import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { pgSearch } from '../../src/eval/pgSource';
import { FakeEmbeddingProvider, fakeEmbedding } from '../../src/llm/fake';
import type { EmbeddingProvider } from '../../src/llm/types';
import { ftsSearch } from '../../src/retrieval/fts';
import { hybridDepth, hybridSearch } from '../../src/retrieval/hybrid';
import { PgRetriever } from '../../src/retrieval/pgRetriever';
import { rrf } from '../../src/retrieval/rrf';
import { vectorSearch } from '../../src/retrieval/vector';
import { insertSyntheticChunks } from './helpers';

const CONFIG = 'md-128-0';
const MODEL = 'nomic-embed-text';
const DIM = 768;
const K = 3;
const DECOYS = 300;
/** camelCase stays one token in the english parser and occurs in no other chunk. */
const IDENTIFIER = 'zeroCopyFlushThreshold';
const TARGET = `Set ${IDENTIFIER} to tune when buffered writes are flushed.`;
const NEIGHBOUR = 'Buffered writes are flushed when the buffer is full.';
const STOP_WORDS = 'the of and';

let sql: Sql;
let targetId: number;
const embedder = new FakeEmbeddingProvider(MODEL, DIM);
/** Proves a search mode makes no embedding call. */
const throwingEmbedder: EmbeddingProvider = {
  model: MODEL,
  dim: DIM,
  embed: () => Promise.reject(new Error('embed must not be called')),
};

const vectorOf = (q: string) => fakeEmbedding(q, DIM);
const ids = (hits: readonly { chunkId: number }[]) => hits.map((h) => h.chunkId);

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  await migrate(sql);
  await ensureEmbeddingTables(sql);
  await sql`TRUNCATE documents, chunk_configs CASCADE`;
  await insertSyntheticChunks(sql, { config: CONFIG, count: DECOYS, sourcePath: 'synthetic/decoys.md' });
  const docIds = await insertSyntheticChunks(sql, {
    config: CONFIG,
    texts: [TARGET, NEIGHBOUR],
    sourcePath: 'guides/flush.md',
  });
  targetId = docIds.get(TARGET) ?? -1;
});

afterAll(async () => {
  await sql?.end();
});

describe('exact identifier', () => {
  it('FTS finds it at rank 1 where vector search misses it, and hybrid keeps it', async () => {
    // Precondition, deterministic: the sha256 fake vector of the query is unrelated to the target's.
    const vector = await vectorSearch(sql, {
      vector: vectorOf(IDENTIFIER),
      k: K,
      chunkConfig: CONFIG,
      model: MODEL,
    });
    expect(vector).toHaveLength(K);
    expect(ids(vector)).not.toContain(targetId);

    const fts = await ftsSearch(sql, { query: IDENTIFIER, k: K, chunkConfig: CONFIG });
    expect(ids(fts)).toEqual([targetId]);

    const hybrid = await hybridSearch(sql, {
      vector: vectorOf(IDENTIFIER),
      query: IDENTIFIER,
      k: K,
      chunkConfig: CONFIG,
      model: MODEL,
    });
    expect(hybrid).toHaveLength(K);
    expect(ids(hybrid)).toContain(targetId);
  });
});

describe('fewer than k and empty FTS', () => {
  it('FTS returns only the matching chunks', async () => {
    const fts = await ftsSearch(sql, { query: 'buffered writes flushed', k: 5, chunkConfig: CONFIG });
    expect(fts).toHaveLength(2);
    expect(fts.every((h, i) => i === 0 || (fts[i - 1]?.rank ?? 0) >= h.rank)).toBe(true);
  });

  it('a stop-word query matches nothing, and hybrid falls back to the vector order', async () => {
    expect(await ftsSearch(sql, { query: STOP_WORDS, k: K, chunkConfig: CONFIG })).toEqual([]);
    const vector = await vectorSearch(sql, {
      vector: vectorOf(STOP_WORDS),
      k: K,
      chunkConfig: CONFIG,
      model: MODEL,
    });
    const hybrid = await hybridSearch(sql, {
      vector: vectorOf(STOP_WORDS),
      query: STOP_WORDS,
      k: K,
      chunkConfig: CONFIG,
      model: MODEL,
    });
    expect(ids(hybrid)).toEqual(ids(vector));
  });
});

describe('SQL fusion agrees with rrf()', () => {
  it.each([IDENTIFIER, 'buffered writes flushed', STOP_WORDS])('for "%s"', async (query) => {
    const depth = hybridDepth(K);
    const vector = await vectorSearch(sql, {
      vector: vectorOf(query),
      k: depth,
      chunkConfig: CONFIG,
      model: MODEL,
    });
    const fts = await ftsSearch(sql, { query, k: depth, chunkConfig: CONFIG });
    const expected = rrf([ids(vector), ids(fts)], K);

    const hybrid = await hybridSearch(sql, {
      vector: vectorOf(query),
      query,
      k: K,
      chunkConfig: CONFIG,
      model: MODEL,
    });
    expect(ids(hybrid)).toEqual(ids(expected));
    hybrid.forEach((h, i) => {
      expect(h.score).toBeCloseTo(expected[i]?.score ?? Number.NaN, 12);
    });
  });
});

describe('search mode wiring', () => {
  it('PgRetriever in fts mode makes no embedding call and returns the FTS hits', async () => {
    const retriever = new PgRetriever({
      sql,
      embedder: throwingEmbedder,
      chunkConfig: CONFIG,
      k: K,
      search: 'fts',
    });
    const docs = await retriever.invoke(IDENTIFIER);
    expect(docs.map((d) => d.metadata.chunk_id)).toEqual([targetId]);
  });

  it('PgRetriever in hybrid mode returns the hybridSearch hits', async () => {
    const retriever = new PgRetriever({ sql, embedder, chunkConfig: CONFIG, k: K, search: 'hybrid' });
    const docs = await retriever.invoke(IDENTIFIER);
    const hybrid = await hybridSearch(sql, {
      vector: vectorOf(IDENTIFIER),
      query: IDENTIFIER,
      k: K,
      chunkConfig: CONFIG,
      model: MODEL,
    });
    expect(docs.map((d) => [d.metadata.chunk_id, d.metadata.score])).toEqual(
      hybrid.map((h) => [h.chunkId, h.score]),
    );
  });

  it('pgSearch in fts mode makes no embedding call', async () => {
    const search = pgSearch(sql, throwingEmbedder, { mode: 'fts', chunkConfig: CONFIG, k: K });
    const [hits] = await search([IDENTIFIER]);
    expect(hits?.map((h) => h.chunkId)).toEqual([targetId]);
  });
});
