// Argument parsing, config and database phase shared by the CLIs that take a question (retrieve, ask).
import { parseArgs } from 'node:util';
import type { Sql } from 'postgres';
import { type Config, loadConfig } from '../config';
import { connect, databaseUrl } from '../db/client';
import { DEFAULT_CHUNK_CONFIG, parseChunkConfigKey } from '../ingest/chunk';
import { isConnectionError } from '../llm/retry';
import { DEFAULT_SEARCH, SEARCH_MODES, type SearchMode } from '../retrieval/search';
import { assertK, requireEmbeddings } from '../retrieval/vector';
import { DB_DOWN } from './errors';

export const DEFAULT_K = 5;

/** Flags shared by retrieve and ask, for their usage lines. */
export const QUESTION_FLAGS = `[--k ${DEFAULT_K}] [--chunk ${DEFAULT_CHUNK_CONFIG}] [--embed <model>] [--search ${SEARCH_MODES.join('|')}]`;

export type QuestionArgs =
  | { help: true }
  | {
      help: false;
      question: string;
      k: number;
      chunk: string;
      embed: string | undefined;
      search: SearchMode;
    };

/** Throws on a usage error (exit 2 in the CLIs). */
export function parseQuestionArgs(argv: string[]): QuestionArgs {
  const { values, positionals } = parseArgs({
    args: argv,
    strict: true,
    allowPositionals: true,
    options: {
      k: { type: 'string', default: String(DEFAULT_K) },
      chunk: { type: 'string', default: DEFAULT_CHUNK_CONFIG },
      embed: { type: 'string' },
      search: { type: 'string', default: DEFAULT_SEARCH },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  if (values.help) {
    return { help: true };
  }
  const question = positionals.join(' ').trim();
  if (!question) {
    throw new Error('Missing question');
  }
  const k = Number(values.k);
  assertK(k, '--k');
  const search = values.search;
  if (!(SEARCH_MODES as readonly string[]).includes(search)) {
    throw new Error(`--search must be one of ${SEARCH_MODES.join(', ')}, got "${search}"`);
  }
  return { help: false, question, k, chunk: values.chunk, embed: values.embed, search: search as SearchMode };
}

export interface RetrievalSetup {
  config: Config;
  chunkConfig: string;
  url: string;
}

/** Config with the --embed override, the validated chunk config key and DATABASE_URL; throws on bad config. */
export function resolveRetrievalConfig(args: { chunk: string; embed: string | undefined }): RetrievalSetup {
  const config = loadConfig({ ...process.env, ...(args.embed && { EMBED_MODEL: args.embed }) });
  const chunkConfig = parseChunkConfigKey(args.chunk).key;
  return { config, chunkConfig, url: databaseUrl() };
}

export interface QuestionCliContext extends RetrievalSetup {
  args: Extract<QuestionArgs, { help: false }>;
  sql: Sql;
}

/**
 * Shared start-up of retrieve and ask: usage errors exit 2, bad config exits 1, then the database phase
 * (requireEmbeddings) runs before `body`, so a missing ingest or a stopped database costs no model call.
 */
export async function runQuestionCli(
  argv: string[],
  usage: string,
  body: (ctx: QuestionCliContext) => Promise<number>,
): Promise<number> {
  let args: QuestionArgs;
  try {
    args = parseQuestionArgs(argv);
  } catch (err) {
    console.error(`${err instanceof Error ? err.message : String(err)}\n${usage}`);
    return 2;
  }
  if (args.help) {
    console.log(usage);
    return 0;
  }
  let setup: RetrievalSetup;
  try {
    setup = resolveRetrievalConfig(args);
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    return 1;
  }
  const sql = connect(setup.url);
  try {
    try {
      await requireEmbeddings(sql, setup.config.embedModel, setup.chunkConfig);
    } catch (err) {
      console.error(isConnectionError(err) ? DB_DOWN : err instanceof Error ? err.message : String(err));
      return 1;
    }
    return await body({ ...setup, args, sql });
  } finally {
    await sql.end();
  }
}
