// bun run eval:retrieval --config experiments/baseline.json [--split dev|test] [--dataset eval/golden.jsonl]
//   [--out results]
// Scores one retrieval config against the golden dataset (docs/PLAN.md M11) and writes
// results/<UTC timestamp>_<experiment>_retrieval.json. Exit 0 on success, 1 on a runtime error (missing ingest,
// corpus drift, database or Ollama down), 2 on a usage error (flags or experiment file).
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { Sql } from 'postgres';
import type { Config } from '../config';
import { connect } from '../db/client';
import { DEFAULT_DATASET, type GoldenEntry, loadDataset, parseDataset, SPLITS } from '../eval/dataset';
import { loadExperiment, type RunConfig, singleConfig } from '../eval/experiment';
import { checkIngest, pgVectorSearch } from '../eval/pgSource';
import {
  buildResult,
  type EmbedModelInfo,
  formatSummary,
  resultFileName,
  runRetrievalEval,
  serializeResult,
} from '../eval/runner';
import { isConnectionError } from '../llm/retry';
import { fetchModelDigest } from '../llm/ollamaTags';
import { createEmbeddingProvider } from '../llm/registry';
import { PostgresLlmCallSink } from '../obs/logger';
import { requireEmbeddings } from '../retrieval/vector';
import { resolveRetrievalConfig } from './common';
import { DB_DOWN, describeError, isDatabaseDown } from './errors';

const DEFAULT_OUT = 'results';
const USAGE =
  `Usage: bun run eval:retrieval --config experiments/baseline.json [--split ${SPLITS.join('|')}] ` +
  `[--dataset ${DEFAULT_DATASET}] [--out ${DEFAULT_OUT}]\n` +
  '  --split  overrides base.split of the experiment file';

export interface EvalArgs {
  command: 'retrieval';
  config: string;
  split: RunConfig['split'] | undefined;
  dataset: string;
  out: string;
}

/** Throws on a usage error; undefined for --help. */
export function parseEvalArgs(argv: string[]): EvalArgs | undefined {
  const { values, positionals } = parseArgs({
    args: argv,
    strict: true,
    allowPositionals: true,
    options: {
      config: { type: 'string' },
      split: { type: 'string' },
      dataset: { type: 'string', default: DEFAULT_DATASET },
      out: { type: 'string', default: DEFAULT_OUT },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  if (values.help) {
    return undefined;
  }
  const [command, ...rest] = positionals;
  if (command !== 'retrieval' || rest.length > 0) {
    throw new Error(
      `Unknown eval command "${positionals.join(' ')}": only "retrieval" exists (answers: M12)`,
    );
  }
  if (!values.config) {
    throw new Error('Missing --config');
  }
  const split = values.split;
  if (split !== undefined && !(SPLITS as readonly string[]).includes(split)) {
    throw new Error(`--split must be one of ${SPLITS.join(', ')}, got "${split}"`);
  }
  return {
    command,
    config: values.config,
    split: split as RunConfig['split'] | undefined,
    dataset: values.dataset,
    out: values.out,
  };
}

function git(args: string[]): string {
  const proc = Bun.spawnSync(['git', ...args], { stderr: 'pipe' });
  if (proc.exitCode !== 0) {
    throw new Error(`git ${args.join(' ')} failed: ${proc.stderr.toString().trim()}`);
  }
  return proc.stdout.toString().trim();
}

/**
 * Uncommitted changes anywhere in the repo, except under `out`: earlier result files there are untracked until
 * committed and would mark every later run dirty.
 */
function gitDirty(out: string): boolean {
  const rel = relative(process.cwd(), resolve(out));
  const inRepo = rel !== '' && !rel.startsWith('..') && !isAbsolute(rel);
  return git(['status', '--porcelain', '--', ':/', ...(inRepo ? [`:(exclude)${rel}`] : [])]) !== '';
}

async function embedModelInfo(config: Config): Promise<EmbedModelInfo> {
  if (config.embedProvider === 'fake') {
    return { provider: 'fake', name: config.embedModel, tag: null, digest: null };
  }
  const { tag, digest } = await fetchModelDigest(config.baseURL, config.embedModel);
  return { provider: config.embedProvider, name: config.embedModel, tag, digest };
}

/** Golden entries, or an error message when the dataset is missing or fails the schema. */
async function readEntries(path: string): Promise<{ entries: GoldenEntry[]; hash: string } | string> {
  const dataset = await loadDataset(path);
  if (!dataset?.text.trim()) {
    return `${path} missing or empty`;
  }
  const { entries, issues } = parseDataset(dataset.text);
  const errors = issues.filter((i) => i.level === 'error');
  if (errors.length > 0) {
    return `${path} has ${errors.length} schema errors: run bun run eval:validate`;
  }
  return { entries, hash: dataset.hash };
}

interface RunContext {
  sql: Sql;
  args: EvalArgs;
  run: RunConfig;
  experimentId: string;
  config: Config;
  chunkConfig: string;
}

async function evaluate({ sql, args, run, experimentId, config, chunkConfig }: RunContext): Promise<number> {
  const startedAt = new Date();
  const data = await readEntries(args.dataset);
  if (typeof data === 'string') {
    console.error(data);
    return 1;
  }
  if (!config.corpusSha) {
    console.error('CORPUS_SHA is not set: copy it from .env.example into .env');
    return 1;
  }
  const corpusSha = config.corpusSha;
  // Checked up front so a name collision does not throw away a finished run; `wx` below still guards the race.
  const path = join(args.out, resultFileName(startedAt, experimentId));
  if (existsSync(path)) {
    console.error(`${path} already exists: start the run again in a second`);
    return 1;
  }
  // Database phase first: a missing ingest or corpus drift costs no model call.
  try {
    await requireEmbeddings(sql, config.embedModel, chunkConfig);
    const sources = data.entries
      .filter((e) => e.split === run.split)
      .flatMap((e) => e.evidence.map((ev) => ev.source));
    await checkIngest(sql, { chunkConfig, corpusSha, sources });
  } catch (err) {
    console.error(isConnectionError(err) ? DB_DOWN : err instanceof Error ? err.message : String(err));
    return 1;
  }
  const runId = `eval-retrieval-${randomUUID()}`;
  try {
    const embed = await embedModelInfo(config);
    const embedder = createEmbeddingProvider(config, new PostgresLlmCallSink(sql, { runId }));
    const result = await runRetrievalEval({
      entries: data.entries,
      split: run.split,
      k: run.k,
      search: pgVectorSearch(sql, embedder, { chunkConfig, k: run.k }),
    });
    const output = buildResult(
      {
        experimentId,
        runId,
        startedAt,
        finishedAt: new Date(),
        gitSha: git(['rev-parse', 'HEAD']),
        gitDirty: gitDirty(args.out),
        datasetSha256: data.hash,
        corpusSha,
        config: run,
        embed,
      },
      result,
    );
    await mkdir(args.out, { recursive: true });
    await writeFile(path, serializeResult(output), { flag: 'wx' });
    console.log(formatSummary(output));
    console.log(`wrote ${path}`);
    return 0;
  } catch (err) {
    console.error(
      (await isDatabaseDown(err, sql))
        ? DB_DOWN
        : describeError(err, { baseURL: config.baseURL, model: config.embedModel }),
    );
    return 1;
  }
}

export async function main(argv: string[]): Promise<number> {
  let args: EvalArgs | undefined;
  let runConfig: RunConfig;
  let experimentId: string;
  try {
    args = parseEvalArgs(argv);
    if (!args) {
      console.log(USAGE);
      return 0;
    }
    const experiment = await loadExperiment(args.config);
    const base = singleConfig(experiment, 'retrieval');
    runConfig = { ...base, split: args.split ?? base.split };
    experimentId = experiment.id;
  } catch (err) {
    console.error(`${err instanceof Error ? err.message : String(err)}\n${USAGE}`);
    return 2;
  }
  let setup: ReturnType<typeof resolveRetrievalConfig>;
  try {
    setup = resolveRetrievalConfig({ chunk: runConfig.chunkConfig, embed: runConfig.embed });
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    return 1;
  }
  const sql = connect(setup.url);
  try {
    return await evaluate({ sql, args, run: runConfig, experimentId, ...setup });
  } finally {
    await sql.end();
  }
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
