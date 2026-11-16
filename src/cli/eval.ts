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
import { answerAll, buildAnswerResult, formatAnswerSummary, judgeAll } from '../eval/answers';
import { DEFAULT_DATASET, type GoldenEntry, loadDataset, parseDataset, SPLITS } from '../eval/dataset';
import {
  type EvalType,
  loadExperiment,
  type RunConfig,
  singleConfig,
  type VaryKey,
} from '../eval/experiment';
import { checkIngest, fetchRunKpis, pgSearch } from '../eval/pgSource';
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
import { expanderFor } from '../rag/expand';
import { activeStages, type ChatStage } from '../rag/pipeline';
import { rerankerFor } from '../rag/rerank';
import { rewriterFor } from '../rag/rewrite';
import { requireEmbeddings } from '../retrieval/vector';
import { answerErrorMessage } from './ask';
import { resolveRetrievalConfig } from './common';
import { DB_DOWN, describeError, isDatabaseDown, retrievalErrorMessage } from './errors';

const DEFAULT_OUT = 'results';
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
 * Uncommitted changes anywhere in the repo, except under `out`, in README.md and in experiments/: earlier result
 * files are untracked until committed, `bun run report` rewrites README.md and a new experiment file is recorded in
 * the result's config anyway; none of them changes a number, but each would mark every later run dirty.
 */
function gitDirty(out: string): boolean {
  const rel = relative(process.cwd(), resolve(out));
  const inRepo = rel !== '' && !rel.startsWith('..') && !isAbsolute(rel);
  const excluded = [
    ':(top,exclude)README.md',
    ':(top,exclude)experiments',
    ...(inRepo ? [`:(exclude)${rel}`] : []),
  ];
  return git(['status', '--porcelain', '--', ':/', ...excluded]) !== '';
}

/** Tag and digest from Ollama; throws when Ollama is down or the model is not pulled. */
export async function modelInfo(provider: ProviderName, baseURL: string, name: string): Promise<ModelInfo> {
  if (provider === 'fake') {
    return { provider, name, tag: null, digest: null };
  }
  const { tag, digest } = await fetchModelDigest(baseURL, name);
  return { provider, name, tag, digest };
}

/** Golden entries, or an error message when the dataset is missing or fails the schema. */
export async function readEntries(path: string): Promise<{ entries: GoldenEntry[]; hash: string } | string> {
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

export interface RunContext {
  sql: Sql;
  args: EvalArgs;
  run: RunConfig;
  experimentId: string;
  /** Matrix runs (bun run experiment): the varied key and its value; null for a single config. */
  variedKey: VaryKey | null;
  variant: string | null;
  config: Config;
  chunkConfig: string;
}

/**
 * The database phase of a run, before any model call: the ingest of `chunkConfig` × embedding model exists and
 * matches CORPUS_SHA and the evidence sources of `split`. Throws with the fix command.
 */
export async function checkRunReady(
  sql: Sql,
  opts: {
    config: Config;
    chunkConfig: string;
    corpusSha: string;
    entries: readonly GoldenEntry[];
    split: string;
  },
): Promise<void> {
  await requireEmbeddings(sql, opts.config.embedModel, opts.chunkConfig);
  const sources = opts.entries
    .filter((e) => e.split === opts.split)
    .flatMap((e) => e.evidence.map((ev) => ev.source));
  await checkIngest(sql, { chunkConfig: opts.chunkConfig, corpusSha: opts.corpusSha, sources });
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
  const path = join(args.out, resultFileName(startedAt, experimentId, args.command, ctx.variant));
  if (existsSync(path)) {
    return `${path} already exists: start the run again in a second`;
  }
  // Database phase first: a missing ingest or corpus drift costs no model call.
  try {
    await checkRunReady(sql, {
      config,
      chunkConfig,
      corpusSha: config.corpusSha,
      entries: data.entries,
      split: run.split,
    });
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
    variedKey: p.variedKey,
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
  const models = await checkModels(config, 'retrieval', run);
  if (!models) {
    return 1;
  }
  const { embed, chat } = models;
  try {
    const sink = new PostgresLlmCallSink(sql, { runId });
    const embedder = createEmbeddingProvider(config, sink);
    if (needsChat(run)) {
      await warmUpChat(config);
    }
    const result = await runRetrievalEval({
      entries: p.entries,
      split: run.split,
      k: run.k,
      search: pgSearch(sql, embedder, {
        mode: run.search,
        chunkConfig,
        fuse: run.expand === 'on',
      }),
      ...(run.rewrite === 'on' && { rewrite: rewriterFor(createChatProvider(config, sink)) }),
      ...(run.expand === 'on' && { expand: expanderFor(createChatProvider(config, sink)) }),
      ...(run.rerank === 'on' && { rerank: rerankerFor(createChatProvider(config, sink)) }),
    });
    const output = buildResult({ ...commonMeta(p, runId), embed, ...(chat && { chat }) }, result);
    await writeResult(p, output, formatSummary(output));
    return 0;
  } catch (err) {
    console.error(await retrievalErrorMessage(err, config, sql));
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
  await warmUpChat(config);
}

/** The chat-model half of the warm-up; also run before a retrieval eval with a chat stage on. */
async function warmUpChat(config: Config): Promise<void> {
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

export type ModelRole = 'embed' | 'chat' | 'judge';

/** A retrieval run calls the chat model when a chat stage is on (rewrite M15, expand M16, rerank M17). */
const needsChat = (run: Pick<RunConfig, ChatStage>): boolean => activeStages(run).length > 0;

/**
 * The models a run of `evalType` calls, with their provider; the judge runs on the chat provider. A retrieval run
 * with a chat stage on calls the chat model too.
 */
export function requiredModels(
  config: Config,
  evalType: EvalType,
  run: Pick<RunConfig, ChatStage>,
): [ModelRole, ProviderName, string][] {
  const embed: [ModelRole, ProviderName, string] = ['embed', config.embedProvider, config.embedModel];
  const chat: [ModelRole, ProviderName, string] = ['chat', config.chatProvider, config.chatModel];
  if (evalType === 'retrieval') {
    return needsChat(run) ? [embed, chat] : [embed];
  }
  return [embed, chat, ['judge', config.chatProvider, config.judgeModel]];
}

/** The checked models of a run: the embedding model always, chat and judge when the run calls them. */
type RunModels = { embed: ModelInfo } & Partial<Record<Exclude<ModelRole, 'embed'>, ModelInfo>>;

/**
 * Every model of the run checked before any work: Ollama down or a model not pulled prints the error naming that
 * model and returns null.
 */
async function checkModels(config: Config, evalType: EvalType, run: RunConfig): Promise<RunModels | null> {
  const found: Partial<Record<ModelRole, ModelInfo>> = {};
  for (const [role, provider, name] of requiredModels(config, evalType, run)) {
    try {
      found[role] = await modelInfo(provider, config.baseURL, name);
    } catch (err) {
      console.error(describeError(err, { baseURL: config.baseURL, model: name }));
      return null;
    }
  }
  const { embed, ...rest } = found;
  if (!embed) {
    throw new Error('requiredModels lists no embedding model');
  }
  return { embed, ...rest };
}

async function evaluateAnswers(p: Prepared): Promise<number> {
  const { sql, run, config, chunkConfig } = p;
  const models = await checkModels(config, 'answers', run);
  if (!models) {
    return 1;
  }
  const { embed, chat, judge } = models;
  if (!chat || !judge) {
    throw new Error('requiredModels lists no chat or judge model for an answers run');
  }
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
            pipeline: { search: run.search, rewrite: run.rewrite, expand: run.expand, rerank: run.rerank },
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

/** One eval run of one config: prelude, then the body; prints its own errors and returns the exit code. */
export async function evaluate(ctx: RunContext): Promise<number> {
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
    return await evaluate({
      sql,
      args,
      run: runConfig,
      experimentId,
      variedKey: null,
      variant: null,
      ...setup,
    });
  } finally {
    await sql.end();
  }
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
