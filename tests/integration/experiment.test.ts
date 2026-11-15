// bun run experiment in process against the test database with fake providers: pre-flight and result files.
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import pgvector from 'pgvector';
import type { Sql } from 'postgres';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, inject, it, vi } from 'vitest';
import { main } from '../../src/cli/experiment';
import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import { parseChunkConfigKey } from '../../src/ingest/chunk';
import { sha256 } from '../../src/ingest/hash';
import { fakeEmbedding } from '../../src/llm/fake';

const CORPUS = 'corpus-sha-experiment';
const SOURCE = 'guides/widget.md';
const TEXT = 'Start the widget with `widget start` on example.com.';

let sql: Sql;
let dir: string;
const errors: string[] = [];

/** One document with one chunk in `config` and its fake nomic embedding. */
async function ingest(config: string): Promise<void> {
  const { chunkSize, chunkOverlap, splitter } = parseChunkConfigKey(config);
  await sql`INSERT INTO chunk_configs (key, chunk_size, chunk_overlap, splitter)
            VALUES (${config}, ${chunkSize}, ${chunkOverlap}, ${splitter}) ON CONFLICT (key) DO NOTHING`;
  const [doc] = await sql<{ id: string }[]>`
    INSERT INTO documents (source_path, title, content_hash, corpus_version)
    VALUES (${SOURCE}, 'Synthetic', ${sha256(SOURCE)}, ${CORPUS})
    ON CONFLICT (source_path) DO UPDATE SET corpus_version = EXCLUDED.corpus_version RETURNING id`;
  const [chunk] = await sql<{ id: string }[]>`
    INSERT INTO chunks (document_id, chunk_config, ordinal, content, content_hash, token_count)
    VALUES (${doc!.id}, ${config}, 0, ${TEXT}, ${sha256(TEXT)}, 10) RETURNING id`;
  await sql`
    INSERT INTO emb_nomic_embed_text (chunk_id, chunk_config, content_hash, embedding)
    VALUES (${chunk!.id}, ${config}, ${sha256(TEXT)}, ${pgvector.toSql(fakeEmbedding(TEXT, 768))})`;
}

async function experimentFile(chunkConfigs: string[]): Promise<string> {
  const path = join(dir, 'exp-chunk-size.json');
  await writeFile(
    path,
    JSON.stringify({
      id: 'exp-chunk-size',
      base: { embed: 'nomic-embed-text', k: 1, search: 'vector', split: 'test' },
      vary: { chunkConfig: chunkConfigs },
      evals: ['retrieval'],
    }),
  );
  return path;
}

const run = (file: string) =>
  main([file, '--dataset', join(dir, 'golden.jsonl'), '--out', join(dir, 'results')]);

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  await migrate(sql);
  await ensureEmbeddingTables(sql);
  await sql`TRUNCATE documents, chunk_configs CASCADE`;
  await ingest('md-512-64');
  await ingest('md-1024-128');
});

afterAll(async () => {
  await sql?.end();
});

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'experiment-'));
  const entry = {
    id: 'q001',
    split: 'test',
    category: 'direct_fact',
    question: TEXT,
    answerable: true,
    evidence: [{ source: SOURCE, span: 'widget start on example.com' }],
    keywords: ['widget'],
    reference_answer: 'Run widget start.',
  };
  await writeFile(join(dir, 'golden.jsonl'), `${JSON.stringify(entry)}\n`);
  vi.stubEnv('DATABASE_URL', inject('databaseUrl'));
  vi.stubEnv('EMBED_PROVIDER', 'fake');
  vi.stubEnv('CHAT_PROVIDER', 'fake');
  vi.stubEnv('CORPUS_SHA', CORPUS);
  errors.length = 0;
  vi.spyOn(console, 'error').mockImplementation((msg: string) => {
    errors.push(msg);
  });
  vi.spyOn(console, 'log').mockImplementation(() => {});
});

afterEach(async () => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  await rm(dir, { recursive: true, force: true });
});

describe('bun run experiment', () => {
  it('pre-flight names every variant without an ingest and runs nothing', async () => {
    const file = await experimentFile(['md-256-32', 'md-512-64', 'md-2048-256']);
    expect(await run(file)).toBe(1);
    const message = errors.join('\n');
    expect(message).toMatch(/pre-flight failed, nothing was run/);
    expect(message).toMatch(
      /No embeddings for md-256-32 \/ nomic-embed-text: run bun run ingest --chunk md-256-32/,
    );
    expect(message).toMatch(/No embeddings for md-2048-256/);
    expect(message).not.toMatch(/md-512-64/);
    await expect(readdir(join(dir, 'results'))).rejects.toThrow();
  });

  it('writes one result file per variant with the variant in its name and header', async () => {
    const file = await experimentFile(['md-512-64', 'md-1024-128']);
    expect(await run(file)).toBe(0);
    const names = (await readdir(join(dir, 'results'))).sort();
    expect(names).toHaveLength(2);
    expect(names[0]).toMatch(/_exp-chunk-size_md-1024-128_retrieval\.json$/);
    expect(names[1]).toMatch(/_exp-chunk-size_md-512-64_retrieval\.json$/);
    for (const name of names) {
      const result = JSON.parse(await readFile(join(dir, 'results', name), 'utf8'));
      expect(result).toMatchObject({
        experimentId: 'exp-chunk-size',
        evalType: 'retrieval',
        variedKey: 'chunkConfig',
      });
      expect(name).toContain(`_${result.config.chunkConfig}_`);
      expect(result.aggregate.all.recall).toBe(1);
    }
  });
});
