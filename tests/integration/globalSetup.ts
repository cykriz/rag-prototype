// Integration tests run against `<db>_test`, never the dev database: schema.test.ts drops the whole schema.
import type { TestProject } from 'vitest/node';
import { connect, databaseUrl } from '../../src/db/client';

declare module 'vitest' {
  export interface ProvidedContext {
    databaseUrl: string;
  }
}

export default async function setup(project: TestProject): Promise<void> {
  // Throws "DATABASE_URL is not set: start the database with `docker compose up -d` …" before any test runs.
  const url = new URL(databaseUrl());
  const name = decodeURIComponent(url.pathname.slice(1));
  if (!name.endsWith('_test')) {
    const testName = `${name}_test`;
    const admin = connect(url.toString());
    try {
      const [row] = await admin<{ exists: boolean }[]>`
        SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = ${testName}) AS exists`;
      if (!row?.exists) {
        await admin`CREATE DATABASE ${admin(testName)}`;
      }
    } finally {
      await admin.end();
    }
    url.pathname = `/${encodeURIComponent(testName)}`;
  }
  project.provide('databaseUrl', url.toString());
}
