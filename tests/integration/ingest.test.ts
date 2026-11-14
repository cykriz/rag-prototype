import { randomUUID } from 'node:crypto';
import { cp, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Sql } from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';

import { type Config, loadConfig } from '../../src/config';
import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { type ChunkConfig, chunkMarkdown, parseChunkConfigKey } from '../../src/ingest/chunk';
import { EMBED_BATCH_SIZE, type IngestOptions, ingest } from '../../src/ingest/ingest';
import { listMarkdownFiles, loadMarkdown } from '../../src/ingest/loadMarkdown';
import { createEmbeddingProvider } from '../../src/llm/registry';
import type { EmbeddingProvider, EmbedResult } from '../../src/llm/types';
import { PostgresLlmCallSink } from '../../src/obs/logger';

const FIXTURES = join(import.meta.dirname, '../fixtures/docs');
const NOMIC = loadConfig({ EMBED_PROVIDER: 'fake' });
const MXBAI = loadConfig({ EMBED_PROVIDER: 'fake', EMBED_MODEL: 'mxbai-embed-large' });
const SMALL = parseChunkConfigKey('md-64-0');
const LARGE = parseChunkConfigKey('md-128-16');
const EDITED = 'cli/commands.md';
const REMOVED = 'changelog.md';

/** Records the size of every embed batch on top of the logged provider. */
class BatchSpy implements EmbeddingProvider {
  readonly batches: number[] = [];
  readonly model: string;
  readonly dim: number;

  constructor(private readonly inner: EmbeddingProvider) {
    this.model = inner.model;
    this.dim = inner.dim;
  }

  embed(texts: string[], kind: 'document' | 'query'): Promise<EmbedResult> {
    this.batches.push(texts.length);
    return this.inner.embed(texts, kind);
  }
}

let sql: Sql;
let root: string;

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  await migrate(sql);
  await ensureEmbeddingTables(sql);
  await sql`TRUNCATE documents, chunk_configs CASCADE`;
  root = await mkdtemp(join(tmpdir(), 'ingest-fixtures-'));
  await cp(FIXTURES, root, { recursive: true });
});

afterAll(async () => {
  await sql?.end();
  if (root) {
    await rm(root, { recursive: true, force: true });
  }
});

async function run(
  opts: { config?: Config; chunkConfig?: ChunkConfig } & Partial<IngestOptions> = {},
): Promise<{ stats: Awaited<ReturnType<typeof ingest>>; batches: number[]; embedRows: number }> {
  const { config = NOMIC, chunkConfig = SMALL, ...rest } = opts;
  const runId = `test-${randomUUID()}`;
  const embedder = new BatchSpy(createEmbeddingProvider(config, new PostgresLlmCallSink(sql, { runId })));
  const stats = await ingest({
    sql,
    root,
    chunkConfig,
    embedder,
    embedding: config.embedding,
    corpusVersion: 'fixture-v1',
    warn: () => {},
    ...rest,
  });
  const [row] = await sql<{ n: number }[]>`
    SELECT count(*)::int AS n FROM llm_calls WHERE run_id = ${runId} AND operation = 'embed'`;
  return { stats, batches: embedder.batches, embedRows: row?.n ?? -1 };
}

/** Expected chunk counts per source path, straight from loader and chunker. */
async function expectedChunks(config: ChunkConfig): Promise<Map<string, number>> {
  const counts = new Map<string, number>();
  for (const path of await listMarkdownFiles(root)) {
    const doc = await loadMarkdown(root, path);
    counts.set(path, (await chunkMarkdown(doc.body, doc.title, config)).length);
  }
  return counts;
}

const sum = (m: Map<string, number>) => [...m.values()].reduce((a, b) => a + b, 0);

async function chunkRows(config: ChunkConfig, path?: string) {
  return sql<{ id: string; source_path: string; ordinal: number; content_hash: string }[]>`
    SELECT c.id, d.source_path, c.ordinal, c.content_hash
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE c.chunk_config = ${config.key} ${path ? sql`AND d.source_path = ${path}` : sql``}
    ORDER BY d.source_path, c.ordinal`;
}

async function documents() {
  return sql<{ source_path: string; ingested_at: Date; corpus_version: string }[]>`
    SELECT source_path, ingested_at, corpus_version FROM documents ORDER BY source_path`;
}

async function embeddingCount(table: string, config: ChunkConfig): Promise<number> {
  const [row] = await sql<{ n: number }[]>`
    SELECT count(*)::int AS n FROM ${sql(table)} WHERE chunk_config = ${config.key}`;
  return row?.n ?? -1;
}

describe('ingest', () => {
  it('rejects a missing or empty corpus directory before touching the database', async () => {
    const empty = await mkdtemp(join(tmpdir(), 'ingest-empty-'));
    try {
      await expect(run({ root: empty })).rejects.toThrow(
        `No .md/.mdx files under ${empty}; run ./scripts/fetch-corpus.sh`,
      );
      await expect(run({ root: join(empty, 'missing') })).rejects.toThrow(
        `Corpus directory ${join(empty, 'missing')} not found; run ./scripts/fetch-corpus.sh`,
      );
    } finally {
      await rm(empty, { recursive: true, force: true });
    }
    expect(await documents()).toEqual([]);
    expect(await sql`SELECT key FROM chunk_configs`).toHaveLength(0);
  });

  it('run 1 ingests every document with batched embeddings', async () => {
    const expected = await expectedChunks(SMALL);
    const total = sum(expected);
    expect(expected.size).toBe(8);
    expect(total).toBeGreaterThan(EMBED_BATCH_SIZE);

    const { stats, batches, embedRows } = await run();

    expect(stats.documents).toMatchObject({ inserted: 8, replaced: 0, skipped: 0 });
    expect(stats.chunksWritten).toBe(total);
    expect(await chunkRows(SMALL)).toHaveLength(total);
    expect(await embeddingCount('emb_nomic_embed_text', SMALL)).toBe(total);
    expect(stats.embeddingsComputed).toBe(total); // all fixture chunks are distinct
    expect(batches).toEqual([EMBED_BATCH_SIZE, total - EMBED_BATCH_SIZE]);
    expect(embedRows).toBe(Math.ceil(total / EMBED_BATCH_SIZE));
    const [config] = await sql`SELECT * FROM chunk_configs`;
    expect(config).toEqual({ key: 'md-64-0', chunk_size: 64, chunk_overlap: 0, splitter: 'markdown' });
  });

  it('run 2 on an unchanged corpus makes zero embedding calls and writes nothing', async () => {
    const chunksBefore = await chunkRows(SMALL);
    const docsBefore = await documents();

    const { stats, embedRows } = await run({ corpusVersion: 'fixture-v2' });

    expect(embedRows).toBe(0);
    expect(stats.documents).toMatchObject({ skipped: 8, inserted: 0, replaced: 0, configAdded: 0 });
    expect(stats.chunksWritten).toBe(0);
    expect(await chunkRows(SMALL)).toEqual(chunksBefore);
    // The blank-body fixture is skipped too, not rewritten; only corpus_version moves.
    expect(await documents()).toEqual(docsBefore.map((d) => ({ ...d, corpus_version: 'fixture-v2' })));
  });

  it('an edited file re-embeds only its changed chunks', async () => {
    const before = await chunkRows(SMALL, EDITED);
    const others = (await chunkRows(SMALL)).filter((c) => c.source_path !== EDITED);
    const path = join(root, EDITED);
    const text = await readFile(path, 'utf8');
    const edited = text.replace(
      'It never changes any files.',
      'It never changes any files and exits with 0.',
    );
    expect(edited).not.toBe(text);
    await writeFile(path, edited);

    const { stats, embedRows } = await run();

    const after = await chunkRows(SMALL, EDITED);
    const oldHashes = new Set(before.map((c) => c.content_hash));
    const changed = after.filter((c) => !oldHashes.has(c.content_hash)).length;
    expect(stats.documents).toMatchObject({ replaced: 1, skipped: 7 });
    expect(changed).toBeGreaterThan(0);
    expect(changed).toBeLessThan(after.length);
    expect(stats.embeddingsComputed).toBe(changed);
    expect(stats.embeddingsWritten).toBe(after.length); // unchanged chunks got their stored vectors
    expect(embedRows).toBe(1);
    expect(await embeddingCount('emb_nomic_embed_text', SMALL)).toBe((await chunkRows(SMALL)).length);
    expect((await chunkRows(SMALL)).filter((c) => c.source_path !== EDITED)).toEqual(others);
  });

  it('a new chunk config adds chunks without touching documents or other configs', async () => {
    const small = await chunkRows(SMALL);
    const docsBefore = await documents();
    const expected = await expectedChunks(LARGE);

    const { stats } = await run({ chunkConfig: LARGE });

    expect(stats.documents).toMatchObject({ configAdded: 7, skipped: 1, replaced: 0, inserted: 0 });
    expect(await chunkRows(LARGE)).toHaveLength(sum(expected));
    expect(await embeddingCount('emb_nomic_embed_text', LARGE)).toBe(sum(expected));
    expect(await chunkRows(SMALL)).toEqual(small);
    expect(await documents()).toEqual(docsBefore);
  });

  it('a new embedding model backfills vectors for the existing chunks', async () => {
    const small = await chunkRows(SMALL);

    const { stats, embedRows } = await run({ config: MXBAI });

    expect(stats.documents).toMatchObject({ backfilled: 7, skipped: 1 });
    expect(stats.chunksWritten).toBe(0);
    expect(stats.embeddingsComputed).toBe(small.length);
    expect(embedRows).toBe(Math.ceil(small.length / EMBED_BATCH_SIZE));
    expect(await embeddingCount('emb_mxbai_embed_large', SMALL)).toBe(small.length);
    expect(await chunkRows(SMALL)).toEqual(small);
    expect((await run({ config: MXBAI })).embedRows).toBe(0);
  });

  it('force re-chunks one config, reuses its stored vectors and leaves other configs alone', async () => {
    const before = await chunkRows(SMALL);
    const large = await chunkRows(LARGE);
    const docsBefore = await documents();

    const { stats, embedRows } = await run({ force: true });

    expect(stats.documents).toMatchObject({ rechunked: 8, replaced: 0, skipped: 0 });
    expect(await documents()).toEqual(docsBefore);
    expect(await chunkRows(LARGE)).toEqual(large);
    expect(await embeddingCount('emb_nomic_embed_text', LARGE)).toBe(large.length);
    expect(stats.embeddingsComputed).toBe(0);
    expect(embedRows).toBe(0);
    const after = await chunkRows(SMALL);
    expect(after.map((c) => c.content_hash)).toEqual(before.map((c) => c.content_hash));
    expect(after.some((c) => before.some((b) => b.id === c.id))).toBe(false);
    expect(await embeddingCount('emb_nomic_embed_text', SMALL)).toBe(after.length);
  });

  it('an edited file drops its chunks of every config; the other config re-adds them on its next run', async () => {
    const path = join(root, 'faq.md');
    await writeFile(path, `${await readFile(path, 'utf8')}\nOne more synthetic sentence.\n`);
    const largeOthers = (await chunkRows(LARGE)).filter((c) => c.source_path !== 'faq.md');

    const small = await run();
    expect(small.stats.documents).toMatchObject({ replaced: 1 });
    expect((await chunkRows(LARGE)).filter((c) => c.source_path === 'faq.md')).toEqual([]);
    expect(await chunkRows(LARGE)).toEqual(largeOthers);

    const large = await run({ chunkConfig: LARGE });
    expect(large.stats.documents).toMatchObject({ configAdded: 1, replaced: 0 });
    expect((await chunkRows(LARGE, 'faq.md')).length).toBeGreaterThan(0);
    expect(await embeddingCount('emb_nomic_embed_text', LARGE)).toBe((await chunkRows(LARGE)).length);
  });

  it('warns once when computed embedding inputs exceed the model context', async () => {
    const warnings: string[] = [];
    const tiny = { ...NOMIC.embedding, contextTokens: 8 };
    await sql`DELETE FROM emb_nomic_embed_text WHERE chunk_config = ${SMALL.key}`;

    const { stats } = await run({ embedding: tiny, warn: (m) => warnings.push(m) });

    expect(stats.oversizedInputs).toBe(stats.embeddingsComputed);
    expect(stats.oversizedInputs).toBeGreaterThan(0);
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(
      new RegExp(`^${stats.oversizedInputs} embedding inputs exceed the model context of 8 tokens`),
    );
    expect((await run({ embedding: tiny, warn: (m) => warnings.push(m) })).stats.oversizedInputs).toBe(0);
    expect(warnings).toHaveLength(1);
  });

  it('a removed file is reported stale, and deleted only with prune', async () => {
    await rm(join(root, REMOVED));

    const kept = await run();
    expect(kept.stats.documents).toMatchObject({ stale: 1, removed: 0 });
    expect((await documents()).map((d) => d.source_path)).toContain(REMOVED);

    const pruned = await run({ prune: true });
    expect(pruned.stats.documents).toMatchObject({ stale: 0, removed: 1 });
    expect((await documents()).map((d) => d.source_path)).not.toContain(REMOVED);
    expect((await chunkRows(SMALL)).filter((c) => c.source_path === REMOVED)).toEqual([]);
    expect(await embeddingCount('emb_nomic_embed_text', SMALL)).toBe((await chunkRows(SMALL)).length);
  });
});
