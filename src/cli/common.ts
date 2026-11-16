// Argument parsing, config and database phase shared by the CLIs that take a question (retrieve, ask).
import { parseArgs } from 'node:util';
import type { Sql } from 'postgres';
import { type Config, loadConfig } from '../config';
import { connect, databaseUrl } from '../db/client';
import { DEFAULT_CHUNK_CONFIG, parseChunkConfigKey } from '../ingest/chunk';
import { isConnectionError } from '../llm/retry';
import {
  assertPipeline,
  DEFAULT_EXPAND,
  DEFAULT_RERANK,
  DEFAULT_REWRITE,
  MAX_TOOL_K,
  type PipelineConfig,
} from '../rag/pipeline';
import { DEFAULT_SEARCH, SEARCH_MODES, type SearchMode } from '../retrieval/search';
import { assertK, requireEmbeddings } from '../retrieval/vector';
import { DB_DOWN } from './errors';

export const DEFAULT_K = 5;

/** Flags shared by retrieve and ask, for their usage lines. */
export const QUESTION_FLAGS = `[--k ${DEFAULT_K}] [--chunk ${DEFAULT_CHUNK_CONFIG}] [--embed <model>] [--search ${SEARCH_MODES.join('|')}] [--rewrite | --expand] [--rerank]`;

/** Values of ask's --pipeline (M18b): the retrieve-then-answer pipeline or the agent. */
export const ANSWER_PIPELINES = ['baseline', 'agent'] as const;

/** Options of parseQuestionArgs/runQuestionCli; only ask takes --pipeline, so retrieve keeps rejecting it. */
export interface QuestionArgsOptions {
  pipelineFlag?: boolean;
}

export type QuestionArgs =
  | { help: true }
  | {
      help: false;
      question: string;
      k: number;
      chunk: string;
      embed: string | undefined;
      /** The stages around retrieval (--search, --rewrite, --expand, --rerank); later stages add their flags here. */
      pipeline: PipelineConfig;
      /** --pipeline agent (ask only, M18b): the agent searches with pipeline.search; the other stages are off. */
      agent: boolean;
    };

/** Throws on a usage error (exit 2 in the CLIs). */
export function parseQuestionArgs(argv: string[], opts: QuestionArgsOptions = {}): QuestionArgs {
  const { values, positionals } = parseArgs({
    args: argv,
    strict: true,
    allowPositionals: true,
    options: {
      k: { type: 'string', default: String(DEFAULT_K) },
      chunk: { type: 'string', default: DEFAULT_CHUNK_CONFIG },
      embed: { type: 'string' },
      search: { type: 'string', default: DEFAULT_SEARCH },
      rewrite: { type: 'boolean', default: DEFAULT_REWRITE === 'on' },
      expand: { type: 'boolean', default: DEFAULT_EXPAND === 'on' },
      rerank: { type: 'boolean', default: DEFAULT_RERANK === 'on' },
      help: { type: 'boolean', short: 'h', default: false },
      ...(opts.pipelineFlag && { pipeline: { type: 'string', default: 'baseline' } }),
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
  const pipeline: PipelineConfig = {
    search: search as SearchMode,
    rewrite: values.rewrite ? 'on' : 'off',
    expand: values.expand ? 'on' : 'off',
    rerank: values.rerank ? 'on' : 'off',
  };
  assertPipeline(pipeline);
  const mode = 'pipeline' in values ? values.pipeline : 'baseline';
  if (!(ANSWER_PIPELINES as readonly unknown[]).includes(mode)) {
    throw new Error(`--pipeline must be one of ${ANSWER_PIPELINES.join(', ')}, got "${mode}"`);
  }
  const agent = mode === 'agent';
  if (agent) {
    // Not evaluated together with the agent; --search stays, it picks the agent's search mode.
    if (values.rewrite || values.expand || values.rerank) {
      throw new Error('--pipeline agent does not combine with --rewrite, --expand or --rerank');
    }
    if (k > MAX_TOOL_K) {
      throw new Error(`--k must be at most ${MAX_TOOL_K} with --pipeline agent, got ${k}`);
    }
  }
  return { help: false, question, k, chunk: values.chunk, embed: values.embed, pipeline, agent };
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
  opts: QuestionArgsOptions = {},
): Promise<number> {
  let args: QuestionArgs;
  try {
    args = parseQuestionArgs(argv, opts);
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
