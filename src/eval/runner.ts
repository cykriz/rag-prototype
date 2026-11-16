// Retrieval eval runner (docs/PLAN.md M11): scores one config against the golden dataset and builds the result
// file of §7. Pure apart from the injected search function, so unit tests need neither database nor model.

import { type Expander, type ExpandRecord, expandRecord, searchQueries } from '../rag/expand';
import { activeStages, assertPipeline, CHAT_STAGES, type ChatStage, retrievalDepth } from '../rag/pipeline';
import { type Reranker, type RerankRecord, rerankItems, rerankRecord } from '../rag/rerank';
import { type RewriteRecord, type Rewriter, rewriteRecord } from '../rag/rewrite';
import type { SearchHit } from '../retrieval/search';
import { CATEGORIES, type Category, type GoldenEntry } from './dataset';
import type { EvalType, RunConfig, VaryKey } from './experiment';
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

/**
 * Stamped on every result file (resultHeader). 2 (M14): retrieval files carry `retrieved[].score` (higher is better)
 * instead of `distance`; the answers layout is unchanged. 3 (M15): `config.rewrite`, `questions[].rewrite`
 * (`{ query, fallback }` or null), `counts.rewriteFallbacks` (null with rewrite off), and answers files carry
 * `latencyP50Ms`/`latencyP95Ms` per `kpis.calls` row. 4 (M16): `config.expand`, `questions[].expand`
 * (`{ queries, dropped, fallback }` or null; `queries` are the deduped expansions searched besides the question),
 * `counts.expandFallbacks` (null with expand off); with expand on, `retrieved[].score` is the RRF sum over the query
 * lists in every search mode. 5 (M17): `config.rerank`, `questions[].rerank` (`{ candidates, order, scores, unknown,
 * duplicates, missing, fallback }` or null; `scores` are the model's, aligned to `order`, null for appended
 * candidates), `counts.rerankFallbacks` (null with rerank off); with rerank on, `retrieved` is in rerank order and
 * `retrieved[].score` stays the search score, so it is no longer descending.
 */
export const RESULT_SCHEMA_VERSION = 5;

/**
 * Top `depth` hits per question, in the order of `queries`: one query list per question (the question alone, or the
 * question and its expansions, M16). runRetrievalEval passes retrievalDepth: k, or the rerank candidates (M17).
 */
export type SearchFn = (queries: string[][], depth: number) => Promise<SearchHit[][]>;

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
  /** The query retrieval used; null without the rewrite stage. */
  rewrite: RewriteRecord | null;
  /** The expansions searched besides the question; null without the expand stage. */
  expand: ExpandRecord | null;
  /** The rerank of the candidates; null without the rerank stage. */
  rerank: RerankRecord | null;
  retrieved: { chunkId: number; sourcePath: string; score: number; covers: number[] }[];
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
export function scoreQuestion(
  entry: GoldenEntry,
  hits: readonly SearchHit[],
  k: number,
  stages: { rewrite?: RewriteRecord | null; expand?: ExpandRecord | null; rerank?: RerankRecord | null } = {},
): QuestionResult {
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
    rewrite: stages.rewrite ?? null,
    expand: stages.expand ?? null,
    rerank: stages.rerank ?? null,
    retrieved: top.map((h, i) => ({
      chunkId: h.chunkId,
      sourcePath: h.sourcePath,
      score: h.score,
      covers: covers[i] ?? [],
    })),
  };
}

export async function runRetrievalEval(opts: {
  entries: readonly GoldenEntry[];
  split: RunConfig['split'];
  k: number;
  search: SearchFn;
  /** Rewrites each question into the search query first (M15), one call per question in id order. */
  rewrite?: Rewriter;
  /** Expands each question into extra queries first (M16), one call per question in id order; not with rewrite. */
  expand?: Expander;
  /**
   * Reranks each question's hits against the question (M17), one call per question in id order, and scores the top
   * k; `search` is asked for retrievalDepth candidates.
   */
  rerank?: Reranker;
}): Promise<RetrievalRun> {
  assertPipeline({ rewrite: opts.rewrite ? 'on' : 'off', expand: opts.expand ? 'on' : 'off' });
  const inSplit = opts.entries.filter((e) => e.split === opts.split);
  const scored = inSplit.filter((e) => e.answerable).sort((a, b) => a.id.localeCompare(b.id));
  const rewrites: (RewriteRecord | null)[] = [];
  const expansions: (ExpandRecord | null)[] = [];
  for (const e of scored) {
    rewrites.push(rewriteRecord(opts.rewrite ? await opts.rewrite(e.question) : null));
    expansions.push(expandRecord(opts.expand ? await opts.expand(e.question) : null));
  }
  const queries = scored.map((e, i) => {
    const expansion = expansions[i] ?? null;
    return expansion ? searchQueries(e.question, expansion) : [rewrites[i]?.query ?? e.question];
  });
  const depth = retrievalDepth({ rerank: opts.rerank ? 'on' : 'off' }, opts.k);
  const searched = scored.length === 0 ? [] : await opts.search(queries, depth);
  if (searched.length !== scored.length) {
    throw new Error(`search returned ${searched.length} hit lists for ${scored.length} questions`);
  }
  const hits: SearchHit[][] = [];
  const reranks: (RerankRecord | null)[] = [];
  for (const [i, e] of scored.entries()) {
    const list = searched[i] ?? [];
    if (opts.rerank) {
      const { kept, trace } = await rerankItems(
        opts.rerank,
        e.question,
        list,
        ({ score: _, ...c }) => c,
        opts.k,
      );
      hits.push(kept);
      reranks.push(rerankRecord(trace));
    } else {
      hits.push(list);
      reranks.push(null);
    }
  }
  return {
    counts: {
      questions: inSplit.length,
      evaluated: scored.length,
      excludedNegative: inSplit.length - scored.length,
    },
    questions: scored.map((e, i) =>
      scoreQuestion(e, hits[i] ?? [], opts.k, {
        rewrite: rewrites[i] ?? null,
        expand: expansions[i] ?? null,
        rerank: reranks[i] ?? null,
      }),
    ),
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
  /** The config key the experiment varies (bun run experiment); null for a single-config run. */
  variedKey: VaryKey | null;
}

export interface ResultMeta extends RunMeta {
  embed: ModelInfo;
  /** The chat model of rewrite, expand or rerank; only with one of them on. */
  chat?: ModelInfo;
}

/** `counts.<stage>Fallbacks` per stage: questions whose stage fell back, null when the run had the stage off. */
export type StageFallbacks = { [S in ChatStage as `${S}Fallbacks`]: number | null };

/** The per-stage fallback counts of a run's rows, keyed as in the result files. */
export function stageFallbacks(
  rows: readonly { [S in ChatStage]: { fallback: boolean } | null }[],
  config: Pick<RunConfig, ChatStage>,
): StageFallbacks {
  const count = (stage: ChatStage) =>
    config[stage] === 'on' ? rows.filter((r) => r[stage]?.fallback === true).length : null;
  // A literal, so the compiler checks every key of StageFallbacks.
  return {
    rewriteFallbacks: count('rewrite'),
    expandFallbacks: count('expand'),
    rerankFallbacks: count('rerank'),
  };
}

/** The stages a run had on, with their fallback counts, for the console summaries. */
export const activeFallbacks = (counts: StageFallbacks): { stage: ChatStage; fallbacks: number }[] =>
  CHAT_STAGES.flatMap((stage) => {
    const fallbacks = counts[`${stage}Fallbacks`];
    return fallbacks === null ? [] : [{ stage, fallbacks }];
  });

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
      rewrite: meta.config.rewrite,
      expand: meta.config.expand,
      rerank: meta.config.rerank,
    },
    variedKey: meta.variedKey,
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
    models: { embed: modelEntry(meta.embed), ...(meta.chat && { chat: modelEntry(meta.chat) }) },
    counts: {
      questions: run.counts.questions,
      evaluated: run.counts.evaluated,
      excludedNegative: run.counts.excludedNegative,
      ...stageFallbacks(run.questions, meta.config),
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
      rewrite: q.rewrite,
      expand: q.expand,
      rerank: q.rerank,
      retrieved: q.retrieved.map((r) => ({
        chunkId: r.chunkId,
        sourcePath: r.sourcePath,
        score: round(r.score, 6),
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

/**
 * `2026-11-15T14-03-12Z_baseline_retrieval.json`, or with a variant `…_exp-k_3_retrieval.json`: UTC to the second,
 * no colons (invalid on some file systems); characters outside `[A-Za-z0-9.-]` in the variant become `-`.
 */
export function resultFileName(
  startedAt: Date,
  experimentId: string,
  evalType: EvalType,
  variant: string | null = null,
): string {
  const stamp = startedAt
    .toISOString()
    .replace(/\.\d{3}Z$/, 'Z')
    .replaceAll(':', '-');
  const middle = variant === null ? '' : `_${variant.replace(/[^A-Za-z0-9.-]/g, '-')}`;
  return `${stamp}_${experimentId}${middle}_${evalType}.json`;
}

/** `md-512-64 / nomic-embed-text, k=5, vector, rerank, test split` for the console summaries; stages only when on. */
export function describeRunConfig(config: RunConfig): string {
  const stages = activeStages(config);
  return [
    `${config.chunkConfig} / ${config.embed}`,
    `k=${config.k}`,
    config.search,
    ...stages,
    `${config.split} split`,
  ].join(', ');
}

const fmt = (x: number | null) => (x === null ? '     -' : x.toFixed(4));

/** Header line plus one row per origin group. */
export function formatSummary(result: RetrievalResult): string {
  const { config, counts, aggregate: agg } = result;
  const k = config.k;
  const lines = [
    `${result.experimentId} (${describeRunConfig(config)}): ` +
      `${counts.evaluated} questions scored, ${counts.excludedNegative} negative excluded` +
      activeFallbacks(counts)
        .map(({ stage, fallbacks }) => `, ${stage} fallback ${fallbacks}/${counts.evaluated}`)
        .join(''),
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
