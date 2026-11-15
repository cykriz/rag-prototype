import pgvector from 'pgvector';
import type { Sql } from 'postgres';
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import { connect } from '../../src/db/client';
import { ensureEmbeddingTables } from '../../src/db/embeddingTables';
import { migrate } from '../../src/db/migrate';
import type { GoldenEntry } from '../../src/eval/dataset';
import { checkIngest, pgVectorSearch } from '../../src/eval/pgSource';
import { runRetrievalEval } from '../../src/eval/runner';
import { sha256 } from '../../src/ingest/hash';
import { FakeEmbeddingProvider, fakeEmbedding } from '../../src/llm/fake';

const CONFIG = 'md-512-64';
const MODEL = 'nomic-embed-text';
const CORPUS = 'corpus-sha-1';
const START = 'Start the widget with `widget start` on example.com.';
const STOP = 'Stop the widget before you edit its config.';

let sql: Sql;

async function insertDocument(sourcePath: string, contents: string[]): Promise<void> {
  const [doc] = await sql<{ id: string }[]>`
    INSERT INTO documents (source_path, title, content_hash, corpus_version)
    VALUES (${sourcePath}, 'Synthetic', ${sha256(sourcePath)}, ${CORPUS}) RETURNING id`;
  const chunks = await sql<{ id: string; content: string }[]>`
    INSERT INTO chunks ${sql(
      contents.map((content, ordinal) => ({
        document_id: doc!.id,
        chunk_config: CONFIG,
        ordinal,
        content,
        content_hash: sha256(content),
        token_count: 10,
      })),
    )} RETURNING id, content`;
  // Embedding of the raw content: a question with the same text gets the same fake vector (distance 0).
  await sql`
    INSERT INTO emb_nomic_embed_text ${sql(
      chunks.map((c) => ({
        chunk_id: c.id,
        chunk_config: CONFIG,
        content_hash: sha256(c.content),
        embedding: pgvector.toSql(fakeEmbedding(c.content, 768)),
      })),
    )}`;
}

const entry = (id: string, question: string, source: string, span: string): GoldenEntry => ({
  id,
  split: 'test',
  category: 'direct_fact',
  question,
  answerable: true,
  evidence: [{ source, span }],
  keywords: ['widget'],
  reference_answer: 'answer',
});

beforeAll(async () => {
  sql = connect(inject('databaseUrl'));
  await migrate(sql);
  await ensureEmbeddingTables(sql);
  await sql`TRUNCATE documents, chunk_configs CASCADE`;
  await sql`INSERT INTO chunk_configs (key, chunk_size, chunk_overlap, splitter) VALUES (${CONFIG}, 512, 64, 'markdown')`;
  await insertDocument('guides/widget.md', [START, STOP]);
  await insertDocument('guides/other.md', ['Unrelated text about example.org.']);
});

afterAll(async () => {
  await sql?.end();
});

describe('checkIngest', () => {
  it('passes when corpus version and sources match', async () => {
    await expect(
      checkIngest(sql, { chunkConfig: CONFIG, corpusSha: CORPUS, sources: ['guides/widget.md'] }),
    ).resolves.toBeUndefined();
  });

  it('refuses chunks from another corpus version', async () => {
    await expect(
      checkIngest(sql, { chunkConfig: CONFIG, corpusSha: 'corpus-sha-2', sources: [] }),
    ).rejects.toThrow(/come from corpus corpus-sha-1, CORPUS_SHA is corpus-sha-2/);
  });

  it('names evidence sources without chunks', async () => {
    await expect(
      checkIngest(sql, {
        chunkConfig: CONFIG,
        corpusSha: CORPUS,
        sources: ['guides/missing.md', 'guides/widget.md'],
      }),
    ).rejects.toThrow(/evidence sources guides\/missing\.md: re-run bun run ingest --chunk md-512-64/);
  });
});

describe('retrieval eval against the database', () => {
  it('scores a hit at rank 1 and a span no chunk covers', async () => {
    const embedder = new FakeEmbeddingProvider(MODEL, 768);
    const run = await runRetrievalEval({
      entries: [
        entry('q001', START, 'guides/widget.md', 'widget start on example.com'),
        entry('q002', STOP, 'guides/widget.md', 'this span is in no chunk'),
      ],
      split: 'test',
      k: 2,
      search: pgVectorSearch(sql, embedder, { chunkConfig: CONFIG, k: 2 }),
    });
    const [q1, q2] = run.questions;
    expect(q1).toMatchObject({ recall: 1, reciprocalRank: 1, ndcg: 1, firstRelevantRank: 1 });
    expect(q1?.retrieved).toHaveLength(2);
    expect(q1?.retrieved[0]?.distance).toBeCloseTo(0, 6);
    expect(q2).toMatchObject({ recall: 0, reciprocalRank: 0, ndcg: 0, spansCovered: [false] });
  });
});
