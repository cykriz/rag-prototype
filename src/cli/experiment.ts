// bun run experiment experiments/exp-chunk-size.json [--split dev|test] [--seed <uint32>] [--dataset eval/golden.jsonl]
//   [--out results]
// Runs every variant of an experiment file (docs/PLAN.md §7, M13a) through the eval code in process: a pre-flight
// for the whole matrix before any model call, then all retrieval runs, then all answers runs, one result file each
// (results/<stamp>_<experiment>_<variant>_<evalType>.json). Stops at the first failing run and keeps the files
// written so far. Exit 0 on success, 1 on a pre-flight or run error, 2 on a usage error.
import { parseArgs } from 'node:util';
import type { Sql } from 'postgres';
import type { ProviderName } from '../config';
import { connect } from '../db/client';
import { DEFAULT_DATASET, SPLITS } from '../eval/dataset';
import {
  type EvalType,
  type ExperimentFile,
  expandExperiment,
  loadExperiment,
  type RunConfig,
  type Variant,
} from '../eval/experiment';
import { isConnectionError } from '../llm/retry';
import { type RetrievalSetup, resolveRetrievalConfig } from './common';
import { checkRunReady, type EvalArgs, evaluate, modelInfo, readEntries, requiredModels } from './eval';
import { DB_DOWN, describeError } from './errors';

const DEFAULT_OUT = 'results';
const USAGE =
  `Usage: bun run experiment <experiment file> [--split ${SPLITS.join('|')}] [--seed <uint32>] ` +
  `[--dataset ${DEFAULT_DATASET}] [--out ${DEFAULT_OUT}]\n` +
  '  --split  overrides base.split of the experiment file (not allowed when vary.split is set)\n' +
  '  --seed   shuffle seed of every answers run (default: one random seed for the whole matrix)';

export interface ExperimentArgs {
  file: string;
  split: RunConfig['split'] | undefined;
  seed: number | undefined;
  dataset: string;
  out: string;
}

/** Throws on a usage error; undefined for --help. */
export function parseExperimentArgs(argv: string[]): ExperimentArgs | undefined {
  const { values, positionals } = parseArgs({
    args: argv,
    strict: true,
    allowPositionals: true,
    options: {
      split: { type: 'string' },
      seed: { type: 'string' },
      dataset: { type: 'string', default: DEFAULT_DATASET },
      out: { type: 'string', default: DEFAULT_OUT },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  if (values.help) {
    return undefined;
  }
  const [file, ...rest] = positionals;
  if (!file || rest.length > 0) {
    throw new Error('Pass exactly one experiment file');
  }
  const split = values.split;
  if (split !== undefined && !(SPLITS as readonly string[]).includes(split)) {
    throw new Error(`--split must be one of ${SPLITS.join(', ')}, got "${split}"`);
  }
  let seed: number | undefined;
  if (values.seed !== undefined) {
    seed = Number(values.seed);
    if (!/^\d+$/.test(values.seed) || seed > 0xffffffff) {
      throw new Error(`--seed must be an integer 0-${0xffffffff}, got "${values.seed}"`);
    }
  }
  return {
    file,
    split: split as RunConfig['split'] | undefined,
    seed,
    dataset: values.dataset,
    out: values.out,
  };
}

export interface PlannedRun {
  evalType: EvalType;
  variant: Variant;
}

/**
 * Every run of the matrix, eval type major (all retrieval runs first: they are cheap and fail early), variants in
 * file order, with the --split override applied. Throws on a usage error.
 */
export function planRuns(exp: ExperimentFile, split: RunConfig['split'] | undefined): PlannedRun[] {
  if (split !== undefined && exp.vary.split !== undefined) {
    throw new Error(`${exp.id}: --split cannot override vary.split`);
  }
  const variants = expandExperiment(exp).map((v) => (split ? { ...v, run: { ...v.run, split } } : v));
  return exp.evals.flatMap((evalType) => variants.map((variant) => ({ evalType, variant })));
}

/** `retrieval md-256-32`, or `retrieval baseline` for a file without vary. */
const runName = (exp: ExperimentFile, r: PlannedRun) => `${r.evalType} ${r.variant.label ?? exp.id}`;

interface VariantSetup extends RetrievalSetup {
  variant: Variant;
}

/**
 * Everything a run checks before its first model call, for every variant at once: dataset, CORPUS_SHA, ingest per
 * chunk config × embedding model, and every model pulled. Returns all problems (empty when ready).
 */
async function preflight(
  sql: Sql,
  setups: readonly VariantSetup[],
  args: ExperimentArgs,
  evalTypes: readonly EvalType[],
): Promise<string[]> {
  const data = await readEntries(args.dataset);
  if (typeof data === 'string') {
    return [data];
  }
  const first = setups[0];
  if (!first?.config.corpusSha) {
    return ['CORPUS_SHA is not set: copy it from .env.example into .env'];
  }
  const corpusSha = first.config.corpusSha;
  const problems: string[] = [];
  for (const s of setups) {
    try {
      await checkRunReady(sql, {
        config: s.config,
        chunkConfig: s.chunkConfig,
        corpusSha,
        entries: data.entries,
        split: s.variant.run.split,
      });
    } catch (err) {
      if (isConnectionError(err)) {
        return [DB_DOWN];
      }
      problems.push(err instanceof Error ? err.message : String(err));
    }
  }
  const models = new Map<string, [ProviderName, string, string]>();
  for (const { config, variant } of setups) {
    for (const evalType of evalTypes) {
      for (const [, provider, name] of requiredModels(config, evalType, variant.run)) {
        models.set(`${provider}/${name}`, [provider, config.baseURL, name]);
      }
    }
  }
  for (const [provider, baseURL, name] of models.values()) {
    try {
      await modelInfo(provider, baseURL, name);
    } catch (err) {
      problems.push(describeError(err, { baseURL, model: name }));
    }
  }
  return [...new Set(problems)];
}

export async function main(argv: string[]): Promise<number> {
  let args: ExperimentArgs | undefined;
  let exp: ExperimentFile;
  let plan: PlannedRun[];
  try {
    args = parseExperimentArgs(argv);
    if (!args) {
      console.log(USAGE);
      return 0;
    }
    exp = await loadExperiment(args.file);
    plan = planRuns(exp, args.split);
  } catch (err) {
    console.error(`${err instanceof Error ? err.message : String(err)}\n${USAGE}`);
    return 2;
  }
  const variants = [...new Set(plan.map((r) => r.variant))];
  let setups: VariantSetup[];
  try {
    setups = variants.map((variant) => ({
      variant,
      ...resolveRetrievalConfig({ chunk: variant.run.chunkConfig, embed: variant.run.embed }),
    }));
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    return 1;
  }
  const first = setups[0];
  if (!first) {
    console.error(`${exp.id}: no config to run`);
    return 1;
  }
  const sql = connect(first.url);
  try {
    const problems = await preflight(sql, setups, args, exp.evals);
    if (problems.length > 0) {
      console.error(
        `${exp.id}: pre-flight failed, nothing was run:\n${problems.map((p) => `- ${p}`).join('\n')}`,
      );
      return 1;
    }
    // One seed for every answers run, so the variants share the judge order.
    const seed = args.seed ?? crypto.getRandomValues(new Uint32Array(1))[0] ?? 0;
    const done: string[] = [];
    for (const [i, r] of plan.entries()) {
      const setup = setups[variants.indexOf(r.variant)];
      if (!setup) {
        throw new Error(`no setup for ${runName(exp, r)}`);
      }
      const evalArgs: EvalArgs = {
        command: r.evalType,
        config: args.file,
        split: args.split,
        dataset: args.dataset,
        out: args.out,
        seed: r.evalType === 'answers' ? seed : undefined,
      };
      const code = await evaluate({
        sql,
        args: evalArgs,
        run: r.variant.run,
        experimentId: exp.id,
        variedKey: r.variant.variedKey,
        variant: r.variant.label,
        config: setup.config,
        chunkConfig: setup.chunkConfig,
      });
      if (code !== 0) {
        const notRun = plan.slice(i + 1).map((n) => runName(exp, n));
        console.error(
          `${exp.id}: ${runName(exp, r)} failed; done: ${done.join(', ') || 'none'}; ` +
            `not run: ${notRun.join(', ') || 'none'}`,
        );
        return 1;
      }
      done.push(runName(exp, r));
    }
    console.log(`${exp.id}: ${done.length} runs done (${done.join(', ')})`);
    return 0;
  } finally {
    await sql.end();
  }
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
