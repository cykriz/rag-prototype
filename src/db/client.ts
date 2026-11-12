// postgres package connection (docs/PLAN.md M4).
import postgres, { type Sql } from 'postgres';
import { loadConfig } from '../config';

const DATABASE_URL_HINT =
  'start the database with `docker compose up -d` and set DATABASE_URL, e.g. ' +
  'postgres://rag:rag@localhost:5433/rag (see .env.example)';

/** The configured DATABASE_URL; throws a clear error when it is unset. */
export function databaseUrl(env: Record<string, string | undefined> = process.env): string {
  const url = loadConfig(env).databaseUrl;
  if (!url) {
    throw new Error(`DATABASE_URL is not set: ${DATABASE_URL_HINT}`);
  }
  return url;
}

/** NOTICE messages (e.g. "already exists, skipping" on idempotent reruns) are dropped; warnings are printed. */
export function connect(url: string): Sql {
  return postgres(url, {
    onnotice: (notice) => {
      if (notice.severity !== 'NOTICE') {
        console.warn(`${notice.severity}: ${notice.message}`);
      }
    },
  });
}
