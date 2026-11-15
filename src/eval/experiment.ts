// Experiment files (docs/PLAN.md §7): `base` settings plus `vary` lists of values for one variable. M11 runs files
// with an empty `vary` (one config); the matrix runner (M13a) expands the others.
import { readFile } from 'node:fs/promises';
import { z } from 'zod';
import { models } from '../config';
import { parseChunkConfigKey } from '../ingest/chunk';
import { MAX_K } from '../retrieval/vector';
import { SPLITS } from './dataset';

export const SEARCH_MODES = ['vector'] as const;
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

export const RunConfig = z.strictObject({
  chunkConfig: ChunkConfigKey,
  embed: Embed,
  k: K,
  search: Search,
  split: Split,
});
export type RunConfig = z.infer<typeof RunConfig>;

export const ExperimentFile = z.strictObject({
  id: z.string().regex(/^[a-z0-9][a-z0-9-]*$/, 'lowercase letters, digits and "-", like exp-chunk-size'),
  base: RunConfig.partial(),
  vary: z
    .strictObject({
      chunkConfig: z.array(ChunkConfigKey).min(1),
      embed: z.array(Embed).min(1),
      k: z.array(K).min(1),
      search: z.array(Search).min(1),
      split: z.array(Split).min(1),
    })
    .partial()
    .default({}),
  evals: z.array(z.enum(EVAL_TYPES)).min(1),
});
export type ExperimentFile = z.infer<typeof ExperimentFile>;

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

/** The one config of a file without `vary` (M11); throws when `vary` is set or `base` is incomplete. */
export function singleConfig(exp: ExperimentFile, evalType: EvalType): RunConfig {
  if (Object.keys(exp.vary).length > 0) {
    throw new Error(`${exp.id}: "vary" is set; matrix runs come with bun run experiment (M13a)`);
  }
  if (!exp.evals.includes(evalType)) {
    throw new Error(`${exp.id}: "evals" does not include "${evalType}"`);
  }
  const parsed = RunConfig.safeParse(exp.base);
  if (!parsed.success) {
    const missing = parsed.error.issues.map((i) => `base.${i.path.join('.')}: ${i.message}`).join('; ');
    throw new Error(`${exp.id}: ${missing}`);
  }
  return parsed.data;
}
