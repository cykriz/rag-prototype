import { describe, expect, it } from 'vitest';

import { renderEmbeddingsSql, tableName } from '../../src/db/embeddingTables';

const TEMPLATE =
  'CREATE TABLE {{table}} (embedding vector({{dim}})); CREATE INDEX {{table}}_hnsw ON {{table}};';

describe('embedding tables', () => {
  it('derives table names from model names', () => {
    expect(tableName('nomic-embed-text')).toBe('emb_nomic_embed_text');
    expect(tableName('mxbai-embed-large')).toBe('emb_mxbai_embed_large');
  });

  it('substitutes every placeholder', () => {
    expect(renderEmbeddingsSql(TEMPLATE, 'emb_nomic_embed_text', 768)).toBe(
      'CREATE TABLE emb_nomic_embed_text (embedding vector(768)); ' +
        'CREATE INDEX emb_nomic_embed_text_hnsw ON emb_nomic_embed_text;',
    );
  });

  it('rejects table names outside ^emb_[a-z0-9_]+$', () => {
    for (const table of ['chunks', 'emb_', 'emb_x; DROP TABLE chunks', 'emb_Upper']) {
      expect(() => renderEmbeddingsSql(TEMPLATE, table, 768)).toThrow(/Invalid embeddings table name/);
    }
  });

  it('rejects dims HNSW cannot index', () => {
    for (const dim of [0, 2001, 768.5]) {
      expect(() => renderEmbeddingsSql(TEMPLATE, 'emb_x', dim)).toThrow(/Invalid embedding dim/);
    }
  });
});
