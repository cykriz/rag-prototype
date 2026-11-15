// LLM-as-a-judge and answer metrics (docs/PLAN.md §6, M12): the judge call with its Zod schema, judge-score,
// abstention and citation metrics, and the seeded shuffle and self-consistency sample.
import { z } from 'zod';
import type { ChatProvider } from '../llm/types';
import { meanOrNull } from './metrics';
import { buildJudgeMessages, JUDGE_SEED, JUDGE_TEMPERATURE, type JudgeInput } from './prompts';
import type { Origin } from './runner';

export const JudgeResult = z.object({
  reasoning: z.string().max(600),
  accuracy: z.number().int().min(1).max(5),
  completeness: z.number().int().min(1).max(5),
  relevance: z.number().int().min(1).max(5),
});
export type JudgeResult = z.infer<typeof JudgeResult>;

const DIMENSIONS = ['accuracy', 'completeness', 'relevance'] as const;
export type Dimension = (typeof DIMENSIONS)[number];

/** A judge reply that failed validation after the repairs is kept as invalid_output, never dropped. */
export type JudgeVerdict =
  | { status: 'ok'; repairs: number; scores: JudgeResult }
  | { status: 'invalid_output'; repairs: number; error: string };

export async function judgeAnswer(
  chat: Pick<ChatProvider, 'judge'>,
  input: JudgeInput,
): Promise<JudgeVerdict> {
  const res = await chat.judge(
    { messages: buildJudgeMessages(input), temperature: JUDGE_TEMPERATURE, seed: JUDGE_SEED },
    JudgeResult,
  );
  return res.status === 'ok'
    ? { status: 'ok', repairs: res.repairs, scores: res.value }
    : { status: 'invalid_output', repairs: res.repairs, error: res.error };
}

/** What the metrics need from one answered golden entry. */
export interface ScoredQuestion {
  origin: Origin;
  answerable: boolean;
  /** AnswerResult.found: null = unusable output after repairs. */
  found: boolean | null;
  /** found=true only: citations non-empty and none outside the context; null otherwise. */
  citationValid: boolean | null;
  /** First judge pass; null when not judged (negative, abstention or found=null). */
  judge: JudgeVerdict | null;
}

/** §6 citation validity for one answer; null unless found=true. */
export function citationValid(res: {
  found: boolean | null;
  citations: readonly unknown[];
  droppedCitations: readonly number[];
}): boolean | null {
  return res.found === true ? res.citations.length > 0 && res.droppedCitations.length === 0 : null;
}

export interface AnswerMetrics {
  nAnswerable: number;
  nNegative: number;
  /** Answers with valid judge scores: the n of the three means. */
  scored: number;
  accuracy: number | null;
  completeness: number | null;
  relevance: number | null;
  /** Negatives with found=false; found=null counts as a failure. */
  abstentionAccuracy: number | null;
  /** Answerable entries with found=false; found=null is not an abstention. */
  falseAbstentionRate: number | null;
  /** Over found=true answers, negatives included. */
  citationValidity: number | null;
  /** found=null over all entries. */
  invalidAnswerRate: number | null;
}

const share = <T>(rows: readonly T[], hit: (r: T) => boolean) => meanOrNull(rows, (r) => (hit(r) ? 1 : 0));

export function answerMetrics(rows: readonly ScoredQuestion[]): AnswerMetrics {
  const answerable = rows.filter((r) => r.answerable);
  const negative = rows.filter((r) => !r.answerable);
  const scores = rows.flatMap((r) => (r.judge?.status === 'ok' ? [r.judge.scores] : []));
  const mean = (d: Dimension) => meanOrNull(scores, (s) => s[d]);
  return {
    nAnswerable: answerable.length,
    nNegative: negative.length,
    scored: scores.length,
    accuracy: mean('accuracy'),
    completeness: mean('completeness'),
    relevance: mean('relevance'),
    abstentionAccuracy: share(negative, (r) => r.found === false),
    falseAbstentionRate: share(answerable, (r) => r.found === false),
    citationValidity: share(
      rows.filter((r) => r.found === true),
      (r) => r.citationValid === true,
    ),
    invalidAnswerRate: share(rows, (r) => r.found === null),
  };
}

export interface SelfConsistency {
  /** Answers judged a second time. */
  sampled: number;
  /** Sampled answers whose repeat failed validation. */
  repeatInvalid: number;
  /** Sampled answers with valid scores in both passes: the denominator of the shares. */
  pairs: number;
  /** Pairs identical in all three dimensions. */
  identicalShare: number | null;
  perDimension: Record<Dimension, number | null>;
}

export function selfConsistency(
  sample: readonly { first: JudgeVerdict | null; repeat: JudgeVerdict | null }[],
): SelfConsistency {
  const pairs = sample.flatMap(({ first, repeat }) =>
    first?.status === 'ok' && repeat?.status === 'ok' ? [[first.scores, repeat.scores] as const] : [],
  );
  const same = (d: Dimension) => share(pairs, ([a, b]) => a[d] === b[d]);
  return {
    sampled: sample.length,
    repeatInvalid: sample.filter((s) => s.repeat?.status === 'invalid_output').length,
    pairs: pairs.length,
    identicalShare: share(pairs, ([a, b]) => DIMENSIONS.every((d) => a[d] === b[d])),
    perDimension: {
      accuracy: same('accuracy'),
      completeness: same('completeness'),
      relevance: same('relevance'),
    },
  };
}

/** mulberry32: a small seeded PRNG, uniform in [0, 1). Same seed, same sequence. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fisher–Yates with mulberry32(seed); returns a new array. */
export function shuffle<T>(items: readonly T[], seed: number): T[] {
  const out = [...items];
  const rand = mulberry32(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const a = out[i];
    const b = out[j];
    // j <= i < out.length, so both exist; the guard narrows for noUncheckedIndexedAccess.
    if (a === undefined || b === undefined) {
      continue;
    }
    out[i] = b;
    out[j] = a;
  }
  return out;
}

/** §6: 10 % of the judged answers, rounded up, at least one when any exist. Integer math avoids 0.1·30 > 3. */
export function sampleSize(judged: number): number {
  return judged === 0 ? 0 : Math.ceil(judged / 10);
}
