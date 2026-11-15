// Retrieval eval runner (docs/PLAN.md M11): scores one config against the golden dataset and builds the result
// file of §7. Pure apart from the injected search function, so unit tests need neither database nor model.
import { CATEGORIES, type Category, type GoldenEntry } from './dataset';
import type { EvalType, RunConfig } from './experiment';
import {
  type AggregateMetrics,
  aggregate,
  firstRelevantRank,
  keywordCoverage,
  ndcgAtK,
  recallAtK,
  reciprocalRank,
} from './metrics';
import { coverage } from './relevance';

export const RESULT_SCHEMA_VERSION = 1;

export interface RetrievedHit {
  chunkId: number;
  sourcePath: string;
  content: string;
  distance: number;
}

/** Top k hits per question, in the order of `questions`. */
export type SearchFn = (questions: string[]) => Promise<RetrievedHit[][]>;

export type Origin = 'human' | 'llm';

/** Entries without `origin` are human-written (§6). */
export const originOf = (entry: GoldenEntry): Origin => entry.origin ?? 'human';

export interface QuestionResult {
  id: string;
  category: Category;
  origin: Origin;
  recall: number;
  reciprocalRank: number;
  ndcg: number;
  keywordCoverage: number;
  firstRelevantRank: number | null;
  /** Per evidence span, in dataset order: covered by any top k chunk. */
  spansCovered: boolean[];
  retrieved: { chunkId: number; sourcePath: string; distance: number; covers: number[] }[];
}

export interface RetrievalCounts {
  /** Entries in the split. */
  questions: number;
  /** Answerable entries, scored. */
  evaluated: number;
  /** Negative entries, excluded from retrieval metrics (§6). */
  excludedNegative: number;
}

export interface RetrievalRun {
  counts: RetrievalCounts;
  /** Sorted by id. */
  questions: QuestionResult[];
}

/** Metrics of one answerable entry from its top k hits. A span no chunk covers lowers recall; it is no error. */
export function scoreQuestion(entry: GoldenEntry, hits: readonly RetrievedHit[], k: number): QuestionResult {
  const top = hits.slice(0, k);
  const covers = coverage(top, entry.evidence);
  const covered = new Set(covers.flat());
  return {
    id: entry.id,
    category: entry.category,
    origin: originOf(entry),
    recall: recallAtK(covers, entry.evidence.length),
    reciprocalRank: reciprocalRank(covers),
    ndcg: ndcgAtK(covers, entry.evidence.length, k),
    keywordCoverage: keywordCoverage(
      entry.keywords,
      top.map((h) => h.content),
    ),
    firstRelevantRank: firstRelevantRank(covers),
    spansCovered: entry.evidence.map((_, i) => covered.has(i)),
    retrieved: top.map((h, i) => ({
      chunkId: h.chunkId,
      sourcePath: h.sourcePath,
      distance: h.distance,
      covers: covers[i] ?? [],
    })),
  };
}

export async function runRetrievalEval(opts: {
  entries: readonly GoldenEntry[];
  split: RunConfig['split'];
  k: number;
  search: SearchFn;
}): Promise<RetrievalRun> {
  const inSplit = opts.entries.filter((e) => e.split === opts.split);
  const scored = inSplit.filter((e) => e.answerable).sort((a, b) => a.id.localeCompare(b.id));
  const hits = scored.length === 0 ? [] : await opts.search(scored.map((e) => e.question));
  if (hits.length !== scored.length) {
    throw new Error(`search returned ${hits.length} hit lists for ${scored.length} questions`);
  }
  return {
    counts: {
      questions: inSplit.length,
      evaluated: scored.length,
      excludedNegative: inSplit.length - scored.length,
    },
    questions: scored.map((e, i) => scoreQuestion(e, hits[i] ?? [], opts.k)),
  };
}

export interface RetrievalAggregate {
  all: AggregateMetrics;
  human: AggregateMetrics;
  llm: AggregateMetrics;
  /** Every answerable category, in §6 order, n = 0 included. */
  byCategory: Partial<Record<Category, AggregateMetrics>>;
}

/** §6: entries tagged origin "llm" are reported separately from the human-written ones. */
export function aggregateRun(questions: readonly QuestionResult[]): RetrievalAggregate {
  const byCategory: Partial<Record<Category, AggregateMetrics>> = {};
  for (const category of CATEGORIES) {
    if (category !== 'negative') {
      byCategory[category] = aggregate(questions.filter((q) => q.category === category));
    }
  }
  return { ...byOrigin(questions, aggregate), byCategory };
}

/** `fn` over all rows, the human-written and the origin "llm" ones (§6 reports them separately). */
export function byOrigin<R extends { origin: Origin }, A>(
  rows: readonly R[],
  fn: (rows: readonly R[]) => A,
): { all: A; human: A; llm: A } {
  return {
    all: fn(rows),
    human: fn(rows.filter((r) => r.origin === 'human')),
    llm: fn(rows.filter((r) => r.origin === 'llm')),
  };
}

export interface ModelInfo {
  provider: string;
  name: string;
  /** Ollama tag (`name:latest`); null for the fake provider. */
  tag: string | null;
  digest: string | null;
}

/** What every result file records about its run (§7). */
export interface RunMeta {
  experimentId: string;
  runId: string;
  startedAt: Date;
  finishedAt: Date;
  gitSha: string;
  gitDirty: boolean;
  datasetSha256: string;
  corpusSha: string;
  config: RunConfig;
}

export interface ResultMeta extends RunMeta {
  embed: ModelInfo;
}

/** The keys every result file starts with, in this order; shared by the retrieval and answers results. */
export function resultHeader<T extends EvalType>(meta: RunMeta, evalType: T) {
  return {
    schemaVersion: RESULT_SCHEMA_VERSION,
    experimentId: meta.experimentId,
    evalType,
    runId: meta.runId,
    startedAt: meta.startedAt.toISOString(),
    finishedAt: meta.finishedAt.toISOString(),
    gitSha: meta.gitSha,
    gitDirty: meta.gitDirty,
    datasetSha256: meta.datasetSha256,
    corpusSha: meta.corpusSha,
    config: {
      chunkConfig: meta.config.chunkConfig,
      embed: meta.config.embed,
      k: meta.config.k,
      search: meta.config.search,
      split: meta.config.split,
    },
  };
}

export const modelEntry = (m: ModelInfo) => ({
  provider: m.provider,
  name: m.name,
  tag: m.tag,
  digest: m.digest,
});

export const round = (x: number, digits = 4) => Number(x.toFixed(digits));
export const roundOrNull = (x: number | null) => (x === null ? null : round(x));

function roundAggregate(a: AggregateMetrics): AggregateMetrics {
  return {
    n: a.n,
    recall: roundOrNull(a.recall),
    mrr: roundOrNull(a.mrr),
    ndcg: roundOrNull(a.ndcg),
    keywordCoverage: roundOrNull(a.keywordCoverage),
  };
}

/**
 * The result file object. Keys are written in this literal order (JSON.stringify keeps insertion order) and
 * metrics are rounded to 4 decimals, so two runs diff line by line.
 */
export function buildResult(meta: ResultMeta, run: RetrievalRun) {
  const agg = aggregateRun(run.questions);
  const byCategory: Record<string, AggregateMetrics> = {};
  for (const [category, a] of Object.entries(agg.byCategory)) {
    byCategory[category] = roundAggregate(a);
  }
  return {
    ...resultHeader(meta, 'retrieval'),
    models: { embed: modelEntry(meta.embed) },
    counts: {
      questions: run.counts.questions,
      evaluated: run.counts.evaluated,
      excludedNegative: run.counts.excludedNegative,
    },
    aggregate: {
      all: roundAggregate(agg.all),
      human: roundAggregate(agg.human),
      llm: roundAggregate(agg.llm),
      byCategory,
    },
    questions: run.questions.map((q) => ({
      id: q.id,
      category: q.category,
      origin: q.origin,
      recall: round(q.recall),
      reciprocalRank: round(q.reciprocalRank),
      ndcg: round(q.ndcg),
      keywordCoverage: round(q.keywordCoverage),
      firstRelevantRank: q.firstRelevantRank,
      spansCovered: q.spansCovered,
      retrieved: q.retrieved.map((r) => ({
        chunkId: r.chunkId,
        sourcePath: r.sourcePath,
        distance: round(r.distance, 6),
        covers: r.covers,
      })),
    })),
  };
}
export type RetrievalResult = ReturnType<typeof buildResult>;

/** 2-space JSON with a trailing newline. */
export function serializeResult(result: object): string {
  return `${JSON.stringify(result, null, 2)}\n`;
}

/** `2026-11-15T14-03-12Z_baseline_retrieval.json`: UTC to the second, no colons (invalid on some file systems). */
export function resultFileName(startedAt: Date, experimentId: string, evalType: EvalType): string {
  const stamp = startedAt
    .toISOString()
    .replace(/\.\d{3}Z$/, 'Z')
    .replaceAll(':', '-');
  return `${stamp}_${experimentId}_${evalType}.json`;
}

const fmt = (x: number | null) => (x === null ? '     -' : x.toFixed(4));

/** Header line plus one row per origin group. */
export function formatSummary(result: RetrievalResult): string {
  const { config, counts, aggregate: agg } = result;
  const k = config.k;
  const lines = [
    `${result.experimentId} (${config.chunkConfig} / ${config.embed}, k=${k}, ${config.search}, ${config.split} split): ` +
      `${counts.evaluated} questions scored, ${counts.excludedNegative} negative excluded`,
    `${'group'.padEnd(6)}${'n'.padStart(4)}  ${[`Recall@${k}`, `MRR@${k}`, `nDCG@${k}`, `KW@${k}`]
      .map((h) => h.padStart(9))
      .join('')}`,
  ];
  for (const group of ['all', 'human', 'llm'] as const) {
    const a = agg[group];
    lines.push(
      `${group.padEnd(6)}${String(a.n).padStart(4)}  ${[a.recall, a.mrr, a.ndcg, a.keywordCoverage]
        .map((x) => fmt(x).padStart(9))
        .join('')}`,
    );
  }
  return lines.join('\n');
}
