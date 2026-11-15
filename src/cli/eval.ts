// bun run eval:retrieval --config experiments/baseline.json [--split dev|test] [--dataset eval/golden.jsonl]
//   [--out results]
// bun run eval:answers  ... same flags ... [--seed <uint32>]
// Scores one config against the golden dataset (retrieval: docs/PLAN.md M11; answers with LLM-as-a-judge: M12) and
// writes results/<UTC timestamp>_<experiment>_<retrieval|answers>.json. Exit 0 on success, 1 on a runtime error
// (missing ingest, corpus drift, database or Ollama down, model not pulled), 2 on a usage error.
import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import type { Sql } from 'postgres';
import { z } from 'zod';
import type { Config, ProviderName } from '../config';
import { connect } from '../db/client';
import { DEFAULT_DATASET, type GoldenEntry, loadDataset, parseDataset, SPLITS } from '../eval/dataset';
import { answerAll, buildAnswerResult, formatAnswerSummary, judgeAll } from '../eval/answers';
import { type EvalType, loadExperiment, type RunConfig, singleConfig } from '../eval/experiment';
import { checkIngest, fetchRunKpis, pgVectorSearch } from '../eval/pgSource';
import {
  buildResult,
  formatSummary,
  type ModelInfo,
  resultFileName,
  runRetrievalEval,
  serializeResult,
} from '../eval/runner';
import { fetchModelDigest } from '../llm/ollamaTags';
import { createChatProvider, createEmbeddingProvider } from '../llm/registry';
import { isConnectionError } from '../llm/retry';
import { PostgresLlmCallSink } from '../obs/logger';
import { answer, buildAnswerDeps } from '../rag/answer';
import { requireEmbeddings } from '../retrieval/vector';
import { answerErrorMessage } from './ask';
import { resolveRetrievalConfig } from './common';
import { DB_DOWN, describeError, isDatabaseDown } from './errors';

const DEFAULT_OUT = 'results';
/** rag_requests.pipeline of the answer eval; phase 2 pipelines get their own names. */
const PIPELINE = 'baseline';
const USAGE =
  `Usage: bun run eval:<retrieval|answers> --config experiments/baseline.json [--split ${SPLITS.join('|')}] ` +
  `[--dataset ${DEFAULT_DATASET}] [--out ${DEFAULT_OUT}] [--seed <uint32>]\n` +
  '  --split  overrides base.split of the experiment file\n' +
  '  --seed   answers only: shuffle seed of the judge order and self-consistency sample (default: random)';

export interface EvalArgs {
  command: EvalType;
  config: string;
  split: RunConfig['split'] | undefined;
  dataset: string;
  out: string;
  /** answers only; undefined = random, stored in the result either way. */
  seed: number | undefined;
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
      seed: { type: 'string' },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  if (values.help) {
    return undefined;
  }
  const [command, ...rest] = positionals;
  if ((command !== 'retrieval' && command !== 'answers') || rest.length > 0) {
    throw new Error(`Unknown eval command "${positionals.join(' ')}": use "retrieval" or "answers"`);
  }
  if (!values.config) {
    throw new Error('Missing --config');
  }
  const split = values.split;
  if (split !== undefined && !(SPLITS as readonly string[]).includes(split)) {
    throw new Error(`--split must be one of ${SPLITS.join(', ')}, got "${split}"`);
  }
  let seed: number | undefined;
  if (values.seed !== undefined) {
    if (command !== 'answers') {
      throw new Error('--seed only applies to the answers command');
    }
    seed = Number(values.seed);
    if (!/^\d+$/.test(values.seed) || seed > 0xffffffff) {
      throw new Error(`--seed must be an integer 0-${0xffffffff}, got "${values.seed}"`);
    }
  }
  return {
    command,
    config: values.config,
    split: split as RunConfig['split'] | undefined,
    dataset: values.dataset,
    out: values.out,
    seed,
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

/** Tag and digest from Ollama; throws when Ollama is down or the model is not pulled. */
async function modelInfo(provider: ProviderName, baseURL: string, name: string): Promise<ModelInfo> {
  if (provider === 'fake') {
    return { provider, name, tag: null, digest: null };
  }
  const { tag, digest } = await fetchModelDigest(baseURL, name);
  return { provider, name, tag, digest };
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

/** What the shared prelude hands to the eval bodies. */
interface Prepared extends RunContext {
  entries: GoldenEntry[];
  datasetSha256: string;
  corpusSha: string;
  startedAt: Date;
  path: string;
}

/**
 * Dataset, CORPUS_SHA, file name collision and the database phase, before any model call; an error message or the
 * prepared run.
 */
async function prepare(ctx: RunContext): Promise<Prepared | string> {
  const { sql, args, run, experimentId, config, chunkConfig } = ctx;
  const startedAt = new Date();
  const data = await readEntries(args.dataset);
  if (typeof data === 'string') {
    return data;
  }
  if (!config.corpusSha) {
    return 'CORPUS_SHA is not set: copy it from .env.example into .env';
  }
  // Checked up front so a name collision does not throw away a finished run; `wx` below still guards the race.
  const path = join(args.out, resultFileName(startedAt, experimentId, args.command));
  if (existsSync(path)) {
    return `${path} already exists: start the run again in a second`;
  }
  // Database phase first: a missing ingest or corpus drift costs no model call.
  try {
    await requireEmbeddings(sql, config.embedModel, chunkConfig);
    const sources = data.entries
      .filter((e) => e.split === run.split)
      .flatMap((e) => e.evidence.map((ev) => ev.source));
    await checkIngest(sql, { chunkConfig, corpusSha: config.corpusSha, sources });
  } catch (err) {
    return isConnectionError(err) ? DB_DOWN : err instanceof Error ? err.message : String(err);
  }
  return {
    ...ctx,
    entries: data.entries,
    datasetSha256: data.hash,
    corpusSha: config.corpusSha,
    startedAt,
    path,
  };
}

function commonMeta(p: Prepared, runId: string) {
  return {
    experimentId: p.experimentId,
    runId,
    startedAt: p.startedAt,
    finishedAt: new Date(),
    gitSha: git(['rev-parse', 'HEAD']),
    gitDirty: gitDirty(p.args.out),
    datasetSha256: p.datasetSha256,
    corpusSha: p.corpusSha,
    config: p.run,
  };
}

async function writeResult(p: Prepared, result: object, summary: string): Promise<void> {
  await mkdir(p.args.out, { recursive: true });
  await writeFile(p.path, serializeResult(result), { flag: 'wx' });
  console.log(summary);
  console.log(`wrote ${p.path}`);
}

async function evaluateRetrieval(p: Prepared): Promise<number> {
  const { sql, run, config, chunkConfig } = p;
  const runId = `eval-retrieval-${randomUUID()}`;
  try {
    const embed = await modelInfo(config.embedProvider, config.baseURL, config.embedModel);
    const embedder = createEmbeddingProvider(config, new PostgresLlmCallSink(sql, { runId }));
    const result = await runRetrievalEval({
      entries: p.entries,
      split: run.split,
      k: run.k,
      search: pgVectorSearch(sql, embedder, { chunkConfig, k: run.k }),
    });
    const output = buildResult({ ...commonMeta(p, runId), embed }, result);
    await writeResult(p, output, formatSummary(output));
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

/**
 * One request per model before the measured phase (§8), on providers without a sink so it stays out of the run's
 * llm_calls; its outcome is ignored. The fake provider needs no warm-up.
 */
async function warmUpAnswering(config: Config): Promise<void> {
  if (config.embedProvider !== 'fake') {
    await createEmbeddingProvider(config).embed(['warm-up'], 'query');
  }
  if (config.chatProvider !== 'fake') {
    await createChatProvider(config).generate({
      messages: [{ role: 'user', content: 'Reply with OK.' }],
      maxTokens: 1,
    });
  }
}

async function warmUpJudge(config: Config): Promise<void> {
  if (config.chatProvider !== 'fake') {
    await createChatProvider(config).judge(
      { messages: [{ role: 'user', content: 'Reply with {"ok": true}.' }] },
      z.object({ ok: z.boolean() }),
    );
  }
}

async function evaluateAnswers(p: Prepared): Promise<number> {
  const { sql, run, config, chunkConfig } = p;
  // Every model checked before any work: Ollama down or a model not pulled stops here.
  const models: Partial<Record<'embed' | 'chat' | 'judge', ModelInfo>> = {};
  const wanted = [
    ['embed', config.embedProvider, config.embedModel],
    ['chat', config.chatProvider, config.chatModel],
    ['judge', config.chatProvider, config.judgeModel],
  ] as const;
  for (const [role, provider, name] of wanted) {
    try {
      models[role] = await modelInfo(provider, config.baseURL, name);
    } catch (err) {
      console.error(describeError(err, { baseURL: config.baseURL, model: name }));
      return 1;
    }
  }
  const { embed, chat, judge } = models as Record<'embed' | 'chat' | 'judge', ModelInfo>;
  const runId = `eval-answers-${randomUUID()}`;
  // A length-1 array always has index 0; `?? 0` only narrows for noUncheckedIndexedAccess.
  const seed = p.args.seed ?? crypto.getRandomValues(new Uint32Array(1))[0] ?? 0;

  let answered: Awaited<ReturnType<typeof answerAll>>;
  try {
    await warmUpAnswering(config);
    answered = await answerAll({
      entries: p.entries,
      split: run.split,
      answer: (entry) =>
        answer(
          entry.question,
          buildAnswerDeps(sql, config, {
            chunkConfig,
            k: run.k,
            runId,
            pipeline: PIPELINE,
            questionId: entry.id,
          }),
        ),
    });
  } catch (err) {
    console.error(await answerErrorMessage(err, config, sql));
    return 1;
  }
  try {
    await warmUpJudge(config);
    const judged = await judgeAll({
      answered,
      judge: createChatProvider(config, new PostgresLlmCallSink(sql, { runId })),
      seed,
    });
    const kpis = await fetchRunKpis(sql, runId);
    const output = buildAnswerResult(
      { ...commonMeta(p, runId), models: { embed, chat, judge } },
      judged,
      kpis,
    );
    await writeResult(p, output, formatAnswerSummary(output));
    return 0;
  } catch (err) {
    console.error(
      (await isDatabaseDown(err, sql))
        ? DB_DOWN
        : describeError(err, { baseURL: config.baseURL, model: config.judgeModel }),
    );
    return 1;
  }
}

async function evaluate(ctx: RunContext): Promise<number> {
  const prepared = await prepare(ctx);
  if (typeof prepared === 'string') {
    console.error(prepared);
    return 1;
  }
  return ctx.args.command === 'answers' ? evaluateAnswers(prepared) : evaluateRetrieval(prepared);
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
    const base = singleConfig(experiment, args.command);
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
