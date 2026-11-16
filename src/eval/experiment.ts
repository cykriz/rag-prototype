// Experiment files (docs/PLAN.md §7): `base` settings plus `vary`, a list of values for at most one variable.
// eval:* runs files with an empty `vary` (one config); bun run experiment (M13a) expands the others.
import { readFile } from 'node:fs/promises';
import { z } from 'zod';
import { models } from '../config';
import { parseChunkConfigKey } from '../ingest/chunk';
import {
  ANSWER_PIPELINES,
  agentConflict,
  DEFAULT_EXPAND,
  DEFAULT_PIPELINE,
  DEFAULT_RERANK,
  DEFAULT_REWRITE,
  EXPAND_MODES,
  RERANK_MODES,
  REWRITE_MODES,
  STAGE_CONFLICT,
  stagesConflict,
} from '../rag/pipeline';
import { SEARCH_MODES } from '../retrieval/search';
import { MAX_K } from '../retrieval/vector';
import { SPLITS } from './dataset';

export const EVAL_TYPES = ['retrieval', 'answers'] as const;
export type EvalType = (typeof EVAL_TYPES)[number];

const ChunkConfigKey = z.string().superRefine((key, ctx) => {
  try {
    parseChunkConfigKey(key);
  } catch (err) {
    ctx.addIssue({ code: 'custom', message: err instanceof Error ? err.message : String(err) });
  }
});
const Embed = z.enum(Object.keys(models.embeddings) as [string, ...string[]]);
const K = z.number().int().min(1).max(MAX_K);
const Search = z.enum(SEARCH_MODES);
const Split = z.enum(SPLITS);
const Rewrite = z.enum(REWRITE_MODES);
const Expand = z.enum(EXPAND_MODES);
const Rerank = z.enum(RERANK_MODES);
const Pipeline = z.enum(ANSWER_PIPELINES);

const RUN_FIELDS = {
  chunkConfig: ChunkConfigKey,
  embed: Embed,
  k: K,
  search: Search,
  split: Split,
  rewrite: Rewrite,
  expand: Expand,
  rerank: Rerank,
  pipeline: Pipeline,
};

/**
 * `rewrite` (M15), `expand` (M16) and `rerank` (M17) default to off and `pipeline` (M19) to baseline, so result
 * files written before them still parse, read as off and baseline. Rewrite and expand both on is refused
 * (assertPipeline); rerank combines with either. The agent refuses every stage and a k above its tool limit
 * (agentConflict); an agent retrieval eval is refused by ExperimentFile.
 */
export const RunConfig = z
  .strictObject({
    ...RUN_FIELDS,
    rewrite: Rewrite.default(DEFAULT_REWRITE),
    expand: Expand.default(DEFAULT_EXPAND),
    rerank: Rerank.default(DEFAULT_RERANK),
    pipeline: Pipeline.default(DEFAULT_PIPELINE),
  })
  .refine((c) => !stagesConflict(c), { message: STAGE_CONFLICT, path: ['expand'] })
  .superRefine((c, ctx) => {
    const conflict = agentConflict(c);
    if (conflict !== null) {
      ctx.addIssue({ code: 'custom', message: conflict, path: ['pipeline'] });
    }
  });
export type RunConfig = z.infer<typeof RunConfig>;

export const VARY_KEYS = [
  'chunkConfig',
  'embed',
  'k',
  'search',
  'split',
  'rewrite',
  'expand',
  'rerank',
  'pipeline',
] as const;
export type VaryKey = (typeof VARY_KEYS)[number];

const unique = <T>(values: readonly T[]) => new Set(values).size === values.length;

export const ExperimentFile = z
  .strictObject({
    id: z.string().regex(/^[a-z0-9][a-z0-9-]*$/, 'lowercase letters, digits and "-", like exp-chunk-size'),
    // Without the rewrite/expand/rerank/pipeline defaults: Zod 4 applies defaults inside partial(), which would set base.rewrite always and
    // trip the base-vs-vary check below. expandExperiment applies the default.
    base: z.strictObject(RUN_FIELDS).partial(),
    vary: z
      .strictObject({
        chunkConfig: z.array(ChunkConfigKey).min(1).refine(unique, 'values must be unique'),
        embed: z.array(Embed).min(1).refine(unique, 'values must be unique'),
        k: z.array(K).min(1).refine(unique, 'values must be unique'),
        search: z.array(Search).min(1).refine(unique, 'values must be unique'),
        split: z.array(Split).min(1).refine(unique, 'values must be unique'),
        rewrite: z.array(Rewrite).min(1).refine(unique, 'values must be unique'),
        expand: z.array(Expand).min(1).refine(unique, 'values must be unique'),
        rerank: z.array(Rerank).min(1).refine(unique, 'values must be unique'),
        pipeline: z.array(Pipeline).min(1).refine(unique, 'values must be unique'),
      })
      .partial()
      .default({}),
    evals: z.array(z.enum(EVAL_TYPES)).min(1),
  })
  .superRefine((exp, ctx) => {
    const keys = Object.keys(exp.vary) as VaryKey[];
    // §7: one variable at a time (greedy); a cross product would need composite variant labels.
    if (keys.length > 1) {
      ctx.addIssue({
        code: 'custom',
        path: ['vary'],
        message: `one variable per file (§7), got ${keys.join(', ')}`,
      });
    }
    // A retrieval eval of the agent is not defined: its retrieval is whatever its tool calls return (M19).
    const agent = exp.base.pipeline === 'agent' || exp.vary.pipeline?.includes('agent') === true;
    if (agent && exp.evals.includes('retrieval')) {
      ctx.addIssue({
        code: 'custom',
        path: ['evals'],
        message: 'the agent pipeline has no retrieval eval: use "evals": ["answers"]',
      });
    }
    for (const key of keys) {
      if (exp.base[key] !== undefined) {
        ctx.addIssue({
          code: 'custom',
          path: ['base', key],
          message: 'is also varied: set it in base or vary',
        });
      }
    }
  });
export type ExperimentFile = z.infer<typeof ExperimentFile>;

/** One config of an experiment; `variedKey`/`label` are null for a file without `vary`. */
export interface Variant {
  run: RunConfig;
  variedKey: VaryKey | null;
  /** The varied value as text (`md-256-32`, `3`); null without `vary`. */
  label: string | null;
}

/** One variant per `vary` value in file order, or the single base config; throws when `base` is incomplete. */
export function expandExperiment(exp: ExperimentFile): Variant[] {
  const entry = Object.entries(exp.vary)[0] as [VaryKey, readonly (string | number)[]] | undefined;
  const values = entry ? entry[1] : [undefined];
  return values.map((value) => {
    const config = entry ? { ...exp.base, [entry[0]]: value } : exp.base;
    const parsed = RunConfig.safeParse(config);
    if (!parsed.success) {
      const missing = parsed.error.issues.map((i) => `base.${i.path.join('.')}: ${i.message}`).join('; ');
      throw new Error(`${exp.id}: ${missing}`);
    }
    return { run: parsed.data, variedKey: entry ? entry[0] : null, label: entry ? String(value) : null };
  });
}

/** Parsed and validated experiment file; throws with every schema issue on one line. */
export async function loadExperiment(path: string): Promise<ExperimentFile> {
  let raw: unknown;
  try {
    raw = JSON.parse(await readFile(path, 'utf8'));
  } catch (err) {
    throw new Error(`${path}: ${err instanceof Error ? err.message : String(err)}`);
  }
  const parsed = ExperimentFile.safeParse(raw);
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `${i.path.join('.') || '(root)'}: ${i.message}`).join('; ');
    throw new Error(`${path}: ${issues}`);
  }
  return parsed.data;
}

/** The one config of a file without `vary` (eval:*); throws when `vary` is set or `base` is incomplete. */
export function singleConfig(exp: ExperimentFile, evalType: EvalType): RunConfig {
  if (Object.keys(exp.vary).length > 0) {
    throw new Error(`${exp.id}: "vary" is set; run the matrix with bun run experiment <file>`);
  }
  if (!exp.evals.includes(evalType)) {
    throw new Error(`${exp.id}: "evals" does not include "${evalType}"`);
  }
  const [variant] = expandExperiment(exp);
  if (!variant) {
    throw new Error(`${exp.id}: no config`);
  }
  return variant.run;
}
