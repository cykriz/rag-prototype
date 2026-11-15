// Retrieval metrics (docs/PLAN.md §6 "Metric definitions"). Inputs are the coverage lists of relevance.ts for the
// top k chunks in rank order: covers[i] holds the evidence span indices chunk i+1 covers.

/** Share of the `spanCount` evidence spans covered by at least one of the top k chunks. */
export function recallAtK(covers: readonly (readonly number[])[], spanCount: number): number {
  if (spanCount === 0) {
    throw new Error('recall needs at least one evidence span');
  }
  return new Set(covers.flat()).size / spanCount;
}

/** Rank (1-based) of the first chunk covering any span, or null when none does. */
export function firstRelevantRank(covers: readonly (readonly number[])[]): number | null {
  const i = covers.findIndex((c) => c.length > 0);
  return i === -1 ? null : i + 1;
}

/** 1/rank of the first relevant chunk, 0 if none; MRR@k is its mean over questions. */
export function reciprocalRank(covers: readonly (readonly number[])[]): number {
  const rank = firstRelevantRank(covers);
  return rank === null ? 0 : 1 / rank;
}

/**
 * nDCG@k with binary gain: chunk i gains 1 only if it covers a span no earlier chunk covered, so a duplicate of an
 * already covered span earns nothing. IDCG@k places one new span at each of the first min(k, spanCount) ranks.
 */
export function ndcgAtK(covers: readonly (readonly number[])[], spanCount: number, k: number): number {
  if (spanCount === 0) {
    throw new Error('nDCG needs at least one evidence span');
  }
  const seen = new Set<number>();
  let dcg = 0;
  covers.slice(0, k).forEach((c, i) => {
    const fresh = c.filter((s) => !seen.has(s));
    if (fresh.length > 0) {
      dcg += 1 / Math.log2(i + 2);
    }
    for (const s of c) {
      seen.add(s);
    }
  });
  let idcg = 0;
  for (let i = 0; i < Math.min(k, spanCount); i++) {
    idcg += 1 / Math.log2(i + 2);
  }
  return dcg / idcg;
}

/**
 * Share of keywords occurring case-insensitively in the top k contents. Contents are joined by a newline, so a
 * keyword never matches across two chunks.
 */
export function keywordCoverage(keywords: readonly string[], contents: readonly string[]): number {
  if (keywords.length === 0) {
    throw new Error('keyword coverage needs at least one keyword');
  }
  const text = contents.join('\n').toLowerCase();
  return keywords.filter((w) => text.includes(w.toLowerCase())).length / keywords.length;
}

export interface QuestionMetrics {
  recall: number;
  reciprocalRank: number;
  ndcg: number;
  keywordCoverage: number;
}

export interface AggregateMetrics {
  n: number;
  recall: number | null;
  mrr: number | null;
  ndcg: number | null;
  keywordCoverage: number | null;
}

/** Means over questions; null metrics for an empty group (e.g. no llm entries in a split). */
export function aggregate(rows: readonly QuestionMetrics[]): AggregateMetrics {
  const mean = (pick: (r: QuestionMetrics) => number) =>
    rows.length === 0 ? null : rows.reduce((sum, r) => sum + pick(r), 0) / rows.length;
  return {
    n: rows.length,
    recall: mean((r) => r.recall),
    mrr: mean((r) => r.reciprocalRank),
    ndcg: mean((r) => r.ndcg),
    keywordCoverage: mean((r) => r.keywordCoverage),
  };
}
