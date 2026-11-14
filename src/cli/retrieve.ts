// bun run retrieve "<question>" [--k 5] [--chunk md-512-64] [--embed <model>]
// Prints the top k chunks with distance, chunk ID and source (docs/PLAN.md M8).
import { randomUUID } from 'node:crypto';
import { parseArgs } from 'node:util';
import type { Sql } from 'postgres';
import type { Document } from '@langchain/core/documents';
import { type Config, loadConfig } from '../config';
import { connect, databaseUrl } from '../db/client';
import { DEFAULT_CHUNK_CONFIG, parseChunkConfigKey } from '../ingest/chunk';
import { createEmbeddingProvider } from '../llm/registry';
import { isConnectionError } from '../llm/retry';
import { PostgresLlmCallSink } from '../obs/logger';
import { type ChunkMetadata, PgRetriever } from '../retrieval/pgRetriever';
import { assertK, requireEmbeddings } from '../retrieval/vector';
import { describeError } from './errors';

const DEFAULT_K = 5;
const PREVIEW_CHARS = 120;

const USAGE = `Usage: bun run retrieve "<question>" [--k ${DEFAULT_K}] [--chunk ${DEFAULT_CHUNK_CONFIG}] [--embed <model>]`;

const DB_DOWN = 'Database not reachable at DATABASE_URL: start it with `docker compose up -d`';

async function databaseReachable(sql: Sql): Promise<boolean> {
  try {
    await sql`SELECT 1`;
    return true;
  } catch {
    return false;
  }
}

/** Whitespace collapsed, cut to `max` characters with an ellipsis. */
export function preview(text: string, max = PREVIEW_CHARS): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat;
}

export function formatHits(
  question: string,
  meta: { chunkConfig: string; model: string; k: number },
  docs: Document<ChunkMetadata>[],
): string {
  const lines = [`Top ${meta.k} for "${question}" (${meta.chunkConfig} / ${meta.model})`];
  const width = String(docs.length).length;
  docs.forEach((d, i) => {
    const { chunk_id, source_path, heading_path, distance } = d.metadata;
    const rank = String(i + 1).padStart(width + 1);
    const source = heading_path ? `${source_path} > ${heading_path}` : source_path;
    lines.push(`${rank}  ${distance.toFixed(4)}  #${chunk_id}  ${source}`);
    lines.push(`${' '.repeat(width + 1)}  ${preview(d.pageContent)}`);
  });
  if (docs.length < meta.k) {
    lines.push(`only ${docs.length} chunks in ${meta.chunkConfig}`);
  }
  return lines.join('\n');
}

function parse(argv: string[]) {
  const { values, positionals } = parseArgs({
    args: argv,
    strict: true,
    allowPositionals: true,
    options: {
      k: { type: 'string', default: String(DEFAULT_K) },
      chunk: { type: 'string', default: DEFAULT_CHUNK_CONFIG },
      embed: { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  if (values.help) {
    return { help: true as const };
  }
  const question = positionals.join(' ').trim();
  if (!question) {
    throw new Error('Missing question');
  }
  const k = Number(values.k);
  assertK(k, '--k');
  return { help: false as const, question, k, chunk: values.chunk, embed: values.embed };
}

async function main(argv: string[]): Promise<number> {
  let args: ReturnType<typeof parse>;
  try {
    args = parse(argv);
  } catch (err) {
    console.error(`${err instanceof Error ? err.message : String(err)}\n${USAGE}`);
    return 2;
  }
  if (args.help) {
    console.log(USAGE);
    return 0;
  }

  let config: Config;
  let url: string;
  let chunkConfig: string;
  try {
    config = loadConfig({ ...process.env, ...(args.embed && { EMBED_MODEL: args.embed }) });
    chunkConfig = parseChunkConfigKey(args.chunk).key;
    url = databaseUrl();
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    return 1;
  }

  const sql = connect(url);
  try {
    // Database phase first: a missing ingest or a stopped database costs no model call.
    try {
      await requireEmbeddings(sql, config.embedModel, chunkConfig);
    } catch (err) {
      console.error(isConnectionError(err) ? DB_DOWN : err instanceof Error ? err.message : String(err));
      return 1;
    }
    try {
      const embedder = createEmbeddingProvider(
        config,
        new PostgresLlmCallSink(sql, { runId: `retrieve-${randomUUID()}` }),
      );
      const retriever = new PgRetriever({ sql, embedder, chunkConfig, k: args.k });
      const docs = await retriever.invoke(args.question);
      console.log(formatHits(args.question, { chunkConfig, model: config.embedModel, k: args.k }, docs));
      return 0;
    } catch (err) {
      // isConnectionError matches a refused Postgres connection too; probe the database before blaming Ollama.
      const dbDown = isConnectionError(err) && !(await databaseReachable(sql));
      console.error(
        dbDown ? DB_DOWN : describeError(err, { baseURL: config.baseURL, model: config.embedModel }),
      );
      return 1;
    }
  } finally {
    await sql.end();
  }
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
