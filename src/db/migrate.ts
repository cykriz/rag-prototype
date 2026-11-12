// bun run db:migrate — applies migrations/*.sql once, in name order (docs/PLAN.md M4). Each file wraps itself
// in BEGIN/COMMIT and records its version in schema_migrations; this runner skips recorded versions.
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Sql } from 'postgres';
import { connect, databaseUrl } from './client';

export const MIGRATIONS_DIR = fileURLToPath(new URL('../../migrations', import.meta.url));

async function appliedVersions(sql: Sql): Promise<Set<string>> {
  const [row] = await sql<{ exists: boolean }[]>`
    SELECT to_regclass('schema_migrations') IS NOT NULL AS exists`;
  if (!row?.exists) {
    return new Set();
  }
  const rows = await sql<{ version: string }[]>`SELECT version FROM schema_migrations`;
  return new Set(rows.map((r) => r.version));
}

/** Applies pending migrations and returns their versions (file names without `.sql`); [] when up to date. */
export async function migrate(sql: Sql, dir: string = MIGRATIONS_DIR): Promise<string[]> {
  const files = (await readdir(dir)).filter((f) => f.endsWith('.sql')).sort();
  const done = await appliedVersions(sql);
  const applied: string[] = [];
  for (const file of files) {
    const version = file.slice(0, -'.sql'.length);
    if (done.has(version)) {
      continue;
    }
    const text = await Bun.file(join(dir, file)).text();
    // A reserved connection: the file runs its own BEGIN/COMMIT, which postgres.js only allows on a leased
    // connection; on a pooled one a failure surfaces as UNSAFE_TRANSACTION instead of the SQL error.
    const conn = await sql.reserve();
    try {
      await conn.unsafe(text);
    } catch (err) {
      await conn.unsafe('ROLLBACK').catch(() => {});
      throw new Error(`Migration ${file} failed: ${err instanceof Error ? err.message : String(err)}`, {
        cause: err,
      });
    } finally {
      conn.release();
    }
    if (!(await appliedVersions(sql)).has(version)) {
      throw new Error(`Migration ${file} did not record version '${version}' in schema_migrations`);
    }
    applied.push(version);
  }
  return applied;
}

if (import.meta.main) {
  try {
    const sql = connect(databaseUrl());
    try {
      const applied = await migrate(sql);
      console.log(applied.length > 0 ? `Applied: ${applied.join(', ')}` : 'Migrations up to date');
    } finally {
      await sql.end();
    }
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  }
}
