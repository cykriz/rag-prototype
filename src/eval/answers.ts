// Answer eval runner (docs/PLAN.md M12): answers every entry of a split, judges the answered ones in a seeded
// shuffled order, re-judges a 10 % sample and builds the result file of §7. Pure apart from the injected answer
// and judge functions, so unit tests need neither database nor model.
import type { AnswerResult } from '../rag/answer';
import type { Category, GoldenEntry } from './dataset';
import type { RunConfig } from './experiment';
import {
  type AnswerMetrics,
  answerMetrics,
  citationValid,
  type JudgeVerdict,
  judgeAnswer,
  type ScoredQuestion,
  type SelfConsistency,
  sampleSize,
  selfConsistency,
  shuffle,
} from './judge';
import { JUDGE_SEED, JUDGE_TEMPERATURE } from './prompts';
import type { ChatProvider } from '../llm/types';
import {
  byOrigin,
  describeRunConfig,
  type ModelInfo,
  modelEntry,
  originOf,
  type RunMeta,
  resultHeader,
  rewriteFallbacks,
  round,
  roundOrNull,
} from './runner';

export interface AnsweredEntry {
  entry: GoldenEntry;
  result: AnswerResult;
}

/** Every entry of the split, sorted by id, answered one after the other. */
export async function answerAll(opts: {
  entries: readonly GoldenEntry[];
  split: RunConfig['split'];
  answer: (entry: GoldenEntry) => Promise<AnswerResult>;
}): Promise<AnsweredEntry[]> {
  const inSplit = opts.entries.filter((e) => e.split === opts.split).sort((a, b) => a.id.localeCompare(b.id));
  const out: AnsweredEntry[] = [];
  for (const entry of inSplit) {
    out.push({ entry, result: await opts.answer(entry) });
  }
  return out;
}

export interface QuestionOutcome extends AnsweredEntry, ScoredQuestion {
  id: string;
  category: Category;
  /** Second verdict for the self-consistency sample; null otherwise. */
  judgeRepeat: JudgeVerdict | null;
}

export interface JudgedRun {
  shuffleSeed: number;
  /** Sorted by id. */
  questions: QuestionOutcome[];
}

/**
 * Judges answerable entries with found=true (negatives and abstentions are covered by the abstention metrics) in
 * an order shuffled with `seed`; then judges the first sampleSize of that order again.
 */
export async function judgeAll(opts: {
  answered: readonly AnsweredEntry[];
  judge: Pick<ChatProvider, 'judge'>;
  seed: number;
}): Promise<JudgedRun> {
  const toJudge = opts.answered.filter((a) => a.entry.answerable && a.result.found === true);
  const order = shuffle(toJudge, opts.seed);
  const run = (a: AnsweredEntry) =>
    judgeAnswer(opts.judge, {
      question: a.entry.question,
      referenceAnswer: a.entry.reference_answer,
      context: a.result.context,
      answer: a.result.answer ?? '',
    });
  const first = new Map<string, JudgeVerdict>();
  for (const a of order) {
    first.set(a.entry.id, await run(a));
  }
  // After the whole first pass, so a repeat never directly follows its first verdict.
  const repeat = new Map<string, JudgeVerdict>();
  for (const a of order.slice(0, sampleSize(order.length))) {
    repeat.set(a.entry.id, await run(a));
  }
  const questions = opts.answered
    .map((a) => ({
      ...a,
      id: a.entry.id,
      category: a.entry.category,
      origin: originOf(a.entry),
      answerable: a.entry.answerable,
      found: a.result.found,
      citationValid: citationValid(a.result),
      judge: first.get(a.entry.id) ?? null,
      judgeRepeat: repeat.get(a.entry.id) ?? null,
    }))
    .sort((a, b) => a.id.localeCompare(b.id));
  return { shuffleSeed: opts.seed, questions };
}

/** One llm_calls group of the run (pgSource.fetchRunKpis). */
interface CallKpis {
  operation: string;
  model: string;
  calls: number;
  avgInputTokens: number | null;
  avgOutputTokens: number | null;
  /** Share of calls whose token counts are tiktoken estimates. */
  estimatedTokenShare: number;
  /** Share of calls with more than one HTTP attempt. */
  retryRate: number;
  /** Share of calls with status invalid_output. */
  invalidOutputRate: number;
  costUsd: number;
  /** percentile_cont over llm_calls.latency_ms of the group (M15: rewrite cost per request). */
  latencyP50Ms: number | null;
  latencyP95Ms: number | null;
}

export interface AnswerResultMeta extends RunMeta {
  models: { embed: ModelInfo; chat: ModelInfo; judge: ModelInfo };
}

/** §8 KPIs of one run (pgSource.fetchRunKpis). */
export interface RunKpis {
  /** percentile_cont over rag_requests.total_latency_ms of the run; null without requests. */
  latencyMs: { p50: number | null; p95: number | null };
  calls: CallKpis[];
}

function roundMetrics(m: AnswerMetrics): AnswerMetrics {
  return {
    nAnswerable: m.nAnswerable,
    nNegative: m.nNegative,
    scored: m.scored,
    accuracy: roundOrNull(m.accuracy),
    completeness: roundOrNull(m.completeness),
    relevance: roundOrNull(m.relevance),
    abstentionAccuracy: roundOrNull(m.abstentionAccuracy),
    falseAbstentionRate: roundOrNull(m.falseAbstentionRate),
    citationValidity: roundOrNull(m.citationValidity),
    invalidAnswerRate: roundOrNull(m.invalidAnswerRate),
  };
}

const verdict = (v: JudgeVerdict | null) =>
  v === null
    ? null
    : v.status === 'ok'
      ? {
          status: v.status,
          repairs: v.repairs,
          reasoning: v.scores.reasoning,
          accuracy: v.scores.accuracy,
          completeness: v.scores.completeness,
          relevance: v.scores.relevance,
        }
      : { status: v.status, repairs: v.repairs, error: v.error };

/**
 * The result file object. Keys are written in this literal order and rates are rounded to 4 decimals, as in the
 * retrieval result (runner.ts buildResult).
 */
export function buildAnswerResult(meta: AnswerResultMeta, run: JudgedRun, kpis: RunKpis) {
  const qs = run.questions;
  const agg = byOrigin(qs, answerMetrics);
  const sc: SelfConsistency = selfConsistency(
    qs
      .filter((q) => q.judgeRepeat !== null)
      .map((q) => ({
        first: q.judge,
        repeat: q.judgeRepeat,
      })),
  );
  return {
    ...resultHeader(meta, 'answers'),
    models: {
      embed: modelEntry(meta.models.embed),
      chat: modelEntry(meta.models.chat),
      judge: modelEntry(meta.models.judge),
    },
    judge: { temperature: JUDGE_TEMPERATURE, seed: JUDGE_SEED },
    shuffleSeed: run.shuffleSeed,
    counts: {
      questions: qs.length,
      answerable: qs.filter((q) => q.entry.answerable).length,
      negative: qs.filter((q) => !q.entry.answerable).length,
      judged: qs.filter((q) => q.judge !== null).length,
      judgeInvalid: qs.filter((q) => q.judge?.status === 'invalid_output').length,
      invalidAnswers: qs.filter((q) => q.result.found === null).length,
      rewriteFallbacks: rewriteFallbacks(
        qs.map((q) => q.result),
        meta.config.rewrite,
      ),
    },
    aggregate: { all: roundMetrics(agg.all), human: roundMetrics(agg.human), llm: roundMetrics(agg.llm) },
    selfConsistency: {
      sampled: sc.sampled,
      repeatInvalid: sc.repeatInvalid,
      pairs: sc.pairs,
      identicalShare: roundOrNull(sc.identicalShare),
      perDimension: {
        accuracy: roundOrNull(sc.perDimension.accuracy),
        completeness: roundOrNull(sc.perDimension.completeness),
        relevance: roundOrNull(sc.perDimension.relevance),
      },
    },
    kpis: {
      latencyMs: { p50: roundOrNull(kpis.latencyMs.p50), p95: roundOrNull(kpis.latencyMs.p95) },
      calls: kpis.calls.map((c) => ({
        operation: c.operation,
        model: c.model,
        calls: c.calls,
        avgInputTokens: roundOrNull(c.avgInputTokens),
        avgOutputTokens: roundOrNull(c.avgOutputTokens),
        estimatedTokenShare: round(c.estimatedTokenShare),
        retryRate: round(c.retryRate),
        invalidOutputRate: round(c.invalidOutputRate),
        costUsd: round(c.costUsd, 6),
        latencyP50Ms: roundOrNull(c.latencyP50Ms),
        latencyP95Ms: roundOrNull(c.latencyP95Ms),
      })),
    },
    questions: qs.map((q) => ({
      id: q.id,
      category: q.category,
      origin: q.origin,
      answerable: q.entry.answerable,
      requestId: q.result.requestId,
      found: q.result.found,
      status: q.result.status,
      repairs: q.result.repairs,
      latencyMs: q.result.latencyMs,
      answer: q.result.answer,
      citations: q.result.citations.map((c) => c.chunkId),
      droppedCitations: q.result.droppedCitations,
      retrievedIds: q.result.retrievedIds,
      rewrite: q.result.rewrite,
      citationValid: q.citationValid,
      judge: verdict(q.judge),
      judgeRepeat: verdict(q.judgeRepeat),
    })),
  };
}
export type AnswerEvalResult = ReturnType<typeof buildAnswerResult>;

const fmt = (x: number | null) => (x === null ? '-' : x.toFixed(2));
const COLUMNS = ['n', 'neg', 'scored', 'Acc', 'Comp', 'Rel', 'Abst', 'FAbst', 'CitV', 'Inv'];

/** Header lines, one row per origin group, then self-consistency and latency. */
export function formatAnswerSummary(result: AnswerEvalResult): string {
  const { config, counts, models, aggregate: agg, selfConsistency: sc, kpis } = result;
  const lines = [
    `${result.experimentId} (${describeRunConfig(config)}; chat ${models.chat.name}, judge ${models.judge.name}): ` +
      `${counts.questions} questions, ${counts.judged} judged, ${counts.judgeInvalid} judge outputs invalid, ` +
      `${counts.invalidAnswers} answers invalid; shuffle seed ${result.shuffleSeed}`,
    `${'group'.padEnd(6)}${COLUMNS.map((c) => c.padStart(7)).join('')}`,
  ];
  for (const group of ['all', 'human', 'llm'] as const) {
    const a = agg[group];
    const cells = [
      String(a.nAnswerable + a.nNegative),
      String(a.nNegative),
      String(a.scored),
      ...[
        a.accuracy,
        a.completeness,
        a.relevance,
        a.abstentionAccuracy,
        a.falseAbstentionRate,
        a.citationValidity,
        a.invalidAnswerRate,
      ].map(fmt),
    ];
    lines.push(`${group.padEnd(6)}${cells.map((c) => c.padStart(7)).join('')}`);
  }
  lines.push(
    `self-consistency: ${sc.sampled} sampled, ${sc.pairs} pairs, identical ${fmt(sc.identicalShare)} ` +
      `(acc ${fmt(sc.perDimension.accuracy)}, comp ${fmt(sc.perDimension.completeness)}, ` +
      `rel ${fmt(sc.perDimension.relevance)})`,
    `latency p50 ${kpis.latencyMs.p50 ?? '-'} ms, p95 ${kpis.latencyMs.p95 ?? '-'} ms`,
  );
  const rewrite = kpis.calls.find((c) => c.operation === 'rewrite');
  if (counts.rewriteFallbacks !== null) {
    lines.push(
      `rewrite: fallback ${counts.rewriteFallbacks}/${counts.questions}, ` +
        `call p50 ${rewrite?.latencyP50Ms ?? '-'} ms, p95 ${rewrite?.latencyP95Ms ?? '-'} ms`,
    );
  }
  return lines.join('\n');
}
