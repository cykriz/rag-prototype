import { describe, expect, it } from 'vitest';
import type { GoldenEntry } from '../../src/eval/dataset';
import type { RunConfig } from '../../src/eval/experiment';
import {
  buildResult,
  formatSummary,
  type RetrievedHit,
  resultFileName,
  runRetrievalEval,
  serializeResult,
} from '../../src/eval/runner';

const entry = (over: Partial<GoldenEntry> & Pick<GoldenEntry, 'id'>): GoldenEntry => ({
  split: 'test',
  category: 'direct_fact',
  question: `question ${over.id}`,
  answerable: true,
  evidence: [{ source: 'guides/widget.md', span: `fact ${over.id}` }],
  keywords: ['widget'],
  reference_answer: 'answer',
  ...over,
});

const ENTRIES: GoldenEntry[] = [
  entry({ id: 'q003', origin: 'llm', category: 'how_to' }),
  entry({ id: 'q001' }),
  entry({ id: 'q002', split: 'dev' }),
  entry({
    id: 'q004',
    category: 'negative',
    answerable: false,
    evidence: [],
    keywords: [],
  }),
];

const hit = (chunkId: number, content: string, sourcePath = 'guides/widget.md'): RetrievedHit => ({
  chunkId,
  sourcePath,
  content,
  score: chunkId / 3,
});

/** q001: its fact at rank 2; q003: nothing relevant. */
const HITS: Record<string, RetrievedHit[]> = {
  'question q001': [hit(1, 'unrelated'), hit(2, 'The fact q001 about the widget.')],
  'question q003': [hit(3, 'fact q003', 'guides/other.md')],
};

const CONFIG: RunConfig = {
  chunkConfig: 'md-512-64',
  embed: 'nomic-embed-text',
  k: 5,
  search: 'vector',
  split: 'test',
  rewrite: 'off',
  expand: 'off',
};

async function run(split: RunConfig['split'] = 'test') {
  const asked: string[][][] = [];
  const result = await runRetrievalEval({
    entries: ENTRIES,
    split,
    k: 5,
    search: async (qs) => {
      asked.push(qs);
      return qs.map(([q]) => (q === undefined ? [] : (HITS[q] ?? [])));
    },
  });
  return { result, asked };
}

describe('runRetrievalEval', () => {
  it('scores answerable entries of the split, sorted by id, and excludes negatives', async () => {
    const { result, asked } = await run();
    expect(asked).toEqual([[['question q001'], ['question q003']]]);
    expect(result.counts).toEqual({ questions: 3, evaluated: 2, excludedNegative: 1 });
    const [q1, q3] = result.questions;
    expect(q1).toMatchObject({
      id: 'q001',
      origin: 'human',
      recall: 1,
      reciprocalRank: 0.5,
      firstRelevantRank: 2,
      keywordCoverage: 1,
      spansCovered: [true],
    });
    expect(q1?.retrieved.map((r) => r.covers)).toEqual([[], [0]]);
    // Right text, wrong source: not relevant; recall 0 is a result, not an error.
    expect(q3).toMatchObject({ id: 'q003', origin: 'llm', recall: 0, ndcg: 0, firstRelevantRank: null });
  });

  it('searches with the rewritten queries and records them per question', async () => {
    const asked: string[][][] = [];
    const rewritten: string[] = [];
    const result = await runRetrievalEval({
      entries: ENTRIES,
      split: 'test',
      k: 5,
      search: async (qs) => {
        asked.push(qs);
        return qs.map(() => []);
      },
      rewrite: async (q) => {
        rewritten.push(q);
        return q === 'question q001'
          ? { query: 'rewritten q001', fallback: false, repairs: 0 }
          : { query: q, fallback: true, repairs: 2 };
      },
    });
    expect(rewritten).toEqual(['question q001', 'question q003']);
    expect(asked).toEqual([[['rewritten q001'], ['question q003']]]);
    expect(result.questions.map((q) => q.rewrite)).toEqual([
      { query: 'rewritten q001', fallback: false },
      { query: 'question q003', fallback: true },
    ]);
    const out = buildResult(
      {
        experimentId: 'exp-rewrite',
        runId: 'r',
        startedAt: new Date(0),
        finishedAt: new Date(0),
        gitSha: 'a'.repeat(40),
        gitDirty: false,
        datasetSha256: 'b'.repeat(64),
        corpusSha: 'c'.repeat(40),
        config: { ...CONFIG, rewrite: 'on' },
        variedKey: 'rewrite',
        embed: { provider: 'fake', name: 'nomic-embed-text', tag: null, digest: null },
        chat: { provider: 'fake', name: 'fake-chat', tag: null, digest: null },
      },
      result,
    );
    expect(out.counts.rewriteFallbacks).toBe(1);
    expect(out.config.rewrite).toBe('on');
    expect(Object.keys(out.models)).toEqual(['embed', 'chat']);
    expect(formatSummary(out).split('\n')[0]).toContain('vector, rewrite, test split');
    expect(formatSummary(out).split('\n')[0]).toContain('rewrite fallback 1/2');
  });

  it('searches the question with its expansions and records them per question', async () => {
    const asked: string[][][] = [];
    const result = await runRetrievalEval({
      entries: ENTRIES,
      split: 'test',
      k: 5,
      search: async (qs) => {
        asked.push(qs);
        return qs.map(() => []);
      },
      expand: async (q) =>
        q === 'question q001'
          ? { queries: ['expansion a', 'expansion b'], dropped: 1, fallback: false, repairs: 0 }
          : { queries: [], dropped: 0, fallback: true, repairs: 2 },
    });
    expect(asked).toEqual([[['question q001', 'expansion a', 'expansion b'], ['question q003']]]);
    expect(result.questions.map((q) => [q.rewrite, q.expand])).toEqual([
      [null, { queries: ['expansion a', 'expansion b'], dropped: 1, fallback: false }],
      [null, { queries: [], dropped: 0, fallback: true }],
    ]);
    const out = buildResult(
      {
        experimentId: 'exp-expand',
        runId: 'r',
        startedAt: new Date(0),
        finishedAt: new Date(0),
        gitSha: 'a'.repeat(40),
        gitDirty: false,
        datasetSha256: 'b'.repeat(64),
        corpusSha: 'c'.repeat(40),
        config: { ...CONFIG, expand: 'on' },
        variedKey: 'expand',
        embed: { provider: 'fake', name: 'nomic-embed-text', tag: null, digest: null },
        chat: { provider: 'fake', name: 'fake-chat', tag: null, digest: null },
      },
      result,
    );
    expect(out.schemaVersion).toBe(4);
    expect(out.counts).toMatchObject({ rewriteFallbacks: null, expandFallbacks: 1 });
    expect(out.config.expand).toBe('on');
    expect(formatSummary(out).split('\n')[0]).toContain('vector, expand, test split');
    expect(formatSummary(out).split('\n')[0]).toContain('expand fallback 1/2');
  });

  it('records no rewrite without the stage', async () => {
    const { result } = await run();
    expect(result.questions.every((q) => q.rewrite === null && q.expand === null)).toBe(true);
  });

  it('does not call search for an empty split', async () => {
    const result = await runRetrievalEval({
      entries: ENTRIES.filter((e) => e.split === 'test'),
      split: 'dev',
      k: 5,
      search: async () => {
        throw new Error('search must not be called');
      },
    });
    expect(result).toEqual({ counts: { questions: 0, evaluated: 0, excludedNegative: 0 }, questions: [] });
  });
});

describe('buildResult', () => {
  const meta = {
    experimentId: 'baseline',
    runId: 'eval-retrieval-test',
    startedAt: new Date('2026-11-15T14:03:12.345Z'),
    finishedAt: new Date('2026-11-15T14:04:00.000Z'),
    gitSha: 'a'.repeat(40),
    gitDirty: false,
    datasetSha256: 'b'.repeat(64),
    corpusSha: 'c'.repeat(40),
    config: CONFIG,
    variedKey: null,
    embed: { provider: 'fake', name: 'nomic-embed-text', tag: null, digest: null },
  };

  it('reports all, human and llm separately, plus every answerable category', async () => {
    const { result } = await run();
    const out = buildResult(meta, result);
    expect(out.aggregate.all).toEqual({ n: 2, recall: 0.5, mrr: 0.25, ndcg: 0.3155, keywordCoverage: 0.5 });
    expect(out.aggregate.human).toMatchObject({ n: 1, recall: 1, mrr: 0.5 });
    expect(out.aggregate.llm).toMatchObject({ n: 1, recall: 0, mrr: 0 });
    expect(Object.keys(out.aggregate.byCategory)).toEqual([
      'direct_fact',
      'how_to',
      'multi_document',
      'vocabulary_mismatch',
    ]);
    expect(out.aggregate.byCategory.multi_document).toEqual({
      n: 0,
      recall: null,
      mrr: null,
      ndcg: null,
      keywordCoverage: null,
    });
  });

  it('writes keys in a fixed order with a trailing newline', async () => {
    const { result } = await run();
    const out = buildResult(meta, result);
    expect(Object.keys(out)).toEqual([
      'schemaVersion',
      'experimentId',
      'evalType',
      'runId',
      'startedAt',
      'finishedAt',
      'gitSha',
      'gitDirty',
      'datasetSha256',
      'corpusSha',
      'config',
      'variedKey',
      'models',
      'counts',
      'aggregate',
      'questions',
    ]);
    expect(Object.keys(out.questions[0] ?? {})).toEqual([
      'id',
      'category',
      'origin',
      'recall',
      'reciprocalRank',
      'ndcg',
      'keywordCoverage',
      'firstRelevantRank',
      'spansCovered',
      'rewrite',
      'expand',
      'retrieved',
    ]);
    expect(out.questions[0]?.retrieved[0]?.score).toBe(0.333333);
    expect(out.counts.rewriteFallbacks).toBeNull();
    expect(out.config.rewrite).toBe('off');
    expect(serializeResult(out).endsWith('}\n')).toBe(true);
  });

  it('prints a summary with one row per origin group', async () => {
    const { result } = await run();
    const lines = formatSummary(buildResult(meta, result)).split('\n');
    expect(lines[0]).toContain('baseline (md-512-64 / nomic-embed-text, k=5, vector, test split)');
    expect(lines[1]).toMatch(/Recall@5\s+MRR@5\s+nDCG@5\s+KW@5/);
    expect(lines.slice(2).map((l) => l.split(/\s+/)[0])).toEqual(['all', 'human', 'llm']);
  });
});

describe('resultFileName', () => {
  it('uses a UTC timestamp to the second without colons', () => {
    expect(resultFileName(new Date('2026-11-15T14:03:12.345Z'), 'baseline', 'retrieval')).toBe(
      '2026-11-15T14-03-12Z_baseline_retrieval.json',
    );
  });

  it('puts a matrix variant before the eval type and replaces characters unsafe in file names', () => {
    const at = new Date('2026-11-15T14:03:12.345Z');
    expect(resultFileName(at, 'exp-k', 'answers', '3')).toBe('2026-11-15T14-03-12Z_exp-k_3_answers.json');
    expect(resultFileName(at, 'exp-chunk-size', 'retrieval', 'md-256-32')).toBe(
      '2026-11-15T14-03-12Z_exp-chunk-size_md-256-32_retrieval.json',
    );
    expect(resultFileName(at, 'exp-x', 'retrieval', 'model:7b/q4')).toBe(
      '2026-11-15T14-03-12Z_exp-x_model-7b-q4_retrieval.json',
    );
  });
});
