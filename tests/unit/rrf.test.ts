import { describe, expect, it } from 'vitest';
import { pipelineName } from '../../src/rag/pipeline';
import { RRF_K, rrf } from '../../src/retrieval/rrf';
import { SEARCH_MODES } from '../../src/retrieval/search';

const s = (...ranks: number[]) => ranks.reduce((sum, r) => sum + 1 / (RRF_K + r), 0);

describe('rrf', () => {
  it('fuses two ranked lists: ids in both outrank ids in one', () => {
    const vector = [10, 20, 30];
    const fts = [30, 40];
    expect(rrf([vector, fts], 10)).toEqual([
      { chunkId: 30, score: s(3, 1) },
      { chunkId: 10, score: s(1) },
      { chunkId: 20, score: s(2) },
      { chunkId: 40, score: s(2) },
    ]);
  });

  it('breaks equal scores by the lower chunk id', () => {
    expect(rrf([[7], [3]], 10).map((h) => h.chunkId)).toEqual([3, 7]);
  });

  it('keeps the order of the other list when one is empty', () => {
    expect(rrf([[5, 1, 9], []], 10).map((h) => h.chunkId)).toEqual([5, 1, 9]);
  });

  it('cuts to the limit', () => {
    expect(
      rrf(
        [
          [1, 2, 3],
          [4, 5],
        ],
        2,
      ).map((h) => h.chunkId),
    ).toEqual([1, 4]);
  });

  it('returns nothing for no lists', () => {
    expect(rrf([], 5)).toEqual([]);
  });
});

describe('pipelineName', () => {
  it('keeps baseline for vector and names the other modes', () => {
    expect(
      SEARCH_MODES.map((search) => pipelineName({ search, rewrite: 'off', expand: 'off', rerank: 'off' })),
    ).toEqual(['baseline', 'fts', 'hybrid']);
  });

  it('appends the rewrite stage to the search mode', () => {
    expect(
      SEARCH_MODES.map((search) => pipelineName({ search, rewrite: 'on', expand: 'off', rerank: 'off' })),
    ).toEqual(['vector+rewrite', 'fts+rewrite', 'hybrid+rewrite']);
  });
});

describe('pipelineName with expand', () => {
  it('appends the expand stage to the search mode', () => {
    expect(
      SEARCH_MODES.map((search) => pipelineName({ search, rewrite: 'off', expand: 'on', rerank: 'off' })),
    ).toEqual(['vector+expand', 'fts+expand', 'hybrid+expand']);
  });
});

describe('pipelineName with rerank', () => {
  it('appends rerank after the query stages', () => {
    expect(
      SEARCH_MODES.map((search) => pipelineName({ search, rewrite: 'off', expand: 'off', rerank: 'on' })),
    ).toEqual(['vector+rerank', 'fts+rerank', 'hybrid+rerank']);
    expect(pipelineName({ search: 'vector', rewrite: 'on', expand: 'off', rerank: 'on' })).toBe(
      'vector+rewrite+rerank',
    );
    expect(pipelineName({ search: 'hybrid', rewrite: 'off', expand: 'on', rerank: 'on' })).toBe(
      'hybrid+expand+rerank',
    );
  });
});
