// bun run db:embeddings — one embeddings table per model in config/models.json, rendered from
// migrations/templates/embeddings.sql.tpl (docs/PLAN.md §4, M4). Idempotent: the template uses IF NOT EXISTS.
import { join } from 'node:path';
import type { Sql } from 'postgres';
import { type EmbeddingModelConfig, models } from '../config';
import { connect, databaseUrl } from './client';
import { MIGRATIONS_DIR } from './migrate';

const TEMPLATE_PATH = join(MIGRATIONS_DIR, 'templates/embeddings.sql.tpl');
const TABLE_NAME = /^emb_[a-z0-9_]+$/;
/** pgvector's HNSW index supports `vector` up to 2000 dimensions. */
const MAX_HNSW_DIM = 2000;

/** `nomic-embed-text` → `emb_nomic_embed_text`. */
export function tableName(model: string): string {
  return `emb_${model.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`;
}

/** Validates both values before they are spliced into DDL, since identifiers and typmods cannot be bound. */
export function renderEmbeddingsSql(template: string, table: string, dim: number): string {
  if (!TABLE_NAME.test(table)) {
    throw new Error(`Invalid embeddings table name '${table}', expected ${TABLE_NAME}`);
  }
  if (!Number.isInteger(dim) || dim < 1 || dim > MAX_HNSW_DIM) {
    throw new Error(`Invalid embedding dim ${dim} for ${table}, expected an integer 1-${MAX_HNSW_DIM}`);
  }
  return template.replaceAll('{{table}}', table).replaceAll('{{dim}}', String(dim));
}

/**
 * Creates the table and indexes per model, then checks that the `embedding` column has the configured dim, so a
 * table left over from a different config aborts here instead of at the first insert. Returns the table names.
 */
export async function ensureEmbeddingTables(
  sql: Sql,
  embeddingModels: Record<string, Pick<EmbeddingModelConfig, 'dim'>> = models.embeddings,
): Promise<string[]> {
  const template = await Bun.file(TEMPLATE_PATH).text();
  const tables: string[] = [];
  for (const [model, { dim }] of Object.entries(embeddingModels)) {
    const table = tableName(model);
    const ddl = renderEmbeddingsSql(template, table, dim);
    await sql.begin((tx) => tx.unsafe(ddl));
    // For pgvector's `vector(N)`, atttypmod is N.
    const [col] = await sql<{ dim: number }[]>`
      SELECT atttypmod AS dim FROM pg_attribute
      WHERE attrelid = ${table}::regclass AND attname = 'embedding' AND NOT attisdropped`;
    if (col?.dim !== dim) {
      throw new Error(
        `${table}.embedding has dim ${col?.dim ?? 'unknown'}, config/models.json says ${dim} for ${model}; ` +
          'drop the table and re-embed',
      );
    }
    tables.push(table);
  }
  return tables;
}

if (import.meta.main) {
  try {
    const sql = connect(databaseUrl());
    try {
      console.log(`Embedding tables ready: ${(await ensureEmbeddingTables(sql)).join(', ')}`);
    } finally {
      await sql.end();
    }
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
}
