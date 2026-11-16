// Experiment files (docs/PLAN.md §7): `base` settings plus `vary`, a list of values for at most one variable.
// eval:* runs files with an empty `vary` (one config); bun run experiment (M13a) expands the others.
import { readFile } from 'node:fs/promises';
import { z } from 'zod';
import { models } from '../config';
import { parseChunkConfigKey } from '../ingest/chunk';
import { DEFAULT_REWRITE, REWRITE_MODES } from '../rag/pipeline';
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

const RUN_FIELDS = {
  chunkConfig: ChunkConfigKey,
  embed: Embed,
  k: K,
  search: Search,
  split: Split,
  rewrite: Rewrite,
};

/** `rewrite` (M15) defaults to off, so result files written before it still parse, read as off. */
export const RunConfig = z.strictObject({ ...RUN_FIELDS, rewrite: Rewrite.default(DEFAULT_REWRITE) });
export type RunConfig = z.infer<typeof RunConfig>;

export const VARY_KEYS = ['chunkConfig', 'embed', 'k', 'search', 'split', 'rewrite'] as const;
export type VaryKey = (typeof VARY_KEYS)[number];

const unique = <T>(values: readonly T[]) => new Set(values).size === values.length;

export const ExperimentFile = z
  .strictObject({
    id: z.string().regex(/^[a-z0-9][a-z0-9-]*$/, 'lowercase letters, digits and "-", like exp-chunk-size'),
    // Without the rewrite default: Zod 4 applies defaults inside partial(), which would set base.rewrite always and
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
