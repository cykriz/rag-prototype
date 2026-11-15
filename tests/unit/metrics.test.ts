import { describe, expect, it } from 'vitest';
import {
  aggregate,
  firstRelevantRank,
  keywordCoverage,
  ndcgAtK,
  recallAtK,
  reciprocalRank,
} from '../../src/eval/metrics';

// covers[i] = evidence span indices chunk at rank i+1 covers (relevance.ts output).
const LOG2_3 = Math.log2(3); // ≈ 1.58496, discount of rank 2

describe('recallAtK', () => {
  it('counts each covered span once', () => {
    // 2 spans, only span 0 covered (twice) → 1/2.
    expect(recallAtK([[], [0], [0]], 2)).toBe(0.5);
    expect(recallAtK([[1], [0]], 2)).toBe(1);
  });

  it('is 0 when no chunk covers a span, which is no error', () => {
    expect(recallAtK([[], [], []], 3)).toBe(0);
  });

  it('rejects an entry without spans', () => {
    expect(() => recallAtK([], 0)).toThrow(/at least one/);
  });
});

describe('reciprocalRank / MRR', () => {
  it('is 1/rank of the first relevant chunk', () => {
    expect(reciprocalRank([[0]])).toBe(1);
    expect(reciprocalRank([[], [], [1]])).toBeCloseTo(1 / 3, 10);
    expect(firstRelevantRank([[], [], [1]])).toBe(3);
  });

  it('is 0 without a relevant chunk', () => {
    expect(reciprocalRank([[], []])).toBe(0);
    expect(firstRelevantRank([[], []])).toBeNull();
  });

  it('averages to MRR over questions', () => {
    // Questions with first hits at rank 1, 2 and none: (1 + 0.5 + 0) / 3 = 0.5.
    const rows = [[[0]], [[], [0]], [[]]].map((covers) => ({
      recall: 0,
      reciprocalRank: reciprocalRank(covers),
      ndcg: 0,
      keywordCoverage: 0,
    }));
    expect(aggregate(rows).mrr).toBeCloseTo(0.5, 10);
  });
});

describe('ndcgAtK', () => {
  it('is 0 without a relevant chunk', () => {
    expect(ndcgAtK([[], [], [], [], []], 2, 5)).toBe(0);
  });

  it('is 1 for one span found at rank 1, duplicates of it earn nothing', () => {
    // rel = [1, 0, 0]: DCG = 1/log2(2) = 1; IDCG over min(3, 1) = 1 rank = 1.
    expect(ndcgAtK([[0], [0], [0]], 1, 3)).toBe(1);
  });

  it('gives no gain to a chunk that only repeats an already covered span', () => {
    // 2 spans, covers [[0], [0], [1]] → rel = [1, 0, 1].
    // DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5 = 1.5
    // IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093 → nDCG = 0.91972
    const expected = 1.5 / (1 + 1 / LOG2_3);
    expect(ndcgAtK([[0], [0], [1]], 2, 3)).toBeCloseTo(expected, 10);
    expect(expected).toBeCloseTo(0.91972, 5);
  });

  it('gives a chunk covering two new spans a gain of 1, not 2', () => {
    // 2 spans, both in the chunk at rank 2: DCG = 1/log2(3) = 0.63093, IDCG = 1.63093 → 0.38685.
    expect(ndcgAtK([[], [0, 1]], 2, 5)).toBeCloseTo(1 / LOG2_3 / (1 + 1 / LOG2_3), 10);
    expect(ndcgAtK([[], [0, 1]], 2, 5)).toBeCloseTo(0.38685, 5);
  });

  it('caps IDCG at k when there are more spans than ranks', () => {
    // 3 spans, k = 2, both ranks bring a new span: DCG = IDCG = 1 + 1/log2(3) → 1.
    expect(ndcgAtK([[0], [2]], 3, 2)).toBe(1);
  });

  it('ignores chunks beyond k', () => {
    expect(ndcgAtK([[], [], [0]], 1, 2)).toBe(0);
  });
});

describe('keywordCoverage', () => {
  it('matches case-insensitively over the concatenated top k', () => {
    // "Widget.serve" and "PORT" occur (in different chunks), "missing" does not → 2/3.
    const contents = ['Call widget.serve() on example.com', 'Set the port variable'];
    expect(keywordCoverage(['Widget.serve', 'PORT', 'missing'], contents)).toBeCloseTo(2 / 3, 10);
  });

  it('never matches across a chunk boundary', () => {
    expect(keywordCoverage(['alpha beta'], ['one alpha', 'beta two'])).toBe(0);
  });

  it('rejects an entry without keywords', () => {
    expect(() => keywordCoverage([], ['x'])).toThrow(/at least one/);
  });
});

describe('aggregate', () => {
  it('returns null metrics for an empty group', () => {
    expect(aggregate([])).toEqual({ n: 0, recall: null, mrr: null, ndcg: null, keywordCoverage: null });
  });
});
