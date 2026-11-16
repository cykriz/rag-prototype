import { describe, expect, it } from 'vitest';
import type { z } from 'zod';
import {
  type AnsweredEntry,
  answerAll,
  buildAnswerResult,
  formatAnswerSummary,
  judgeAll,
} from '../../src/eval/answers';
import type { GoldenEntry } from '../../src/eval/dataset';
import type { RunConfig } from '../../src/eval/experiment';
import {
  answerMetrics,
  type JudgeVerdict,
  judgeAnswer,
  sampleSize,
  selfConsistency,
  shuffle,
} from '../../src/eval/judge';
import { buildJudgeMessages, JUDGE_SEED, JUDGE_TEMPERATURE } from '../../src/eval/prompts';
import { FakeChatProvider } from '../../src/llm/fake';
import { generateObject } from '../../src/llm/structured';
import type { ChatProvider, GenerateRequest } from '../../src/llm/types';
import type { AnswerResult } from '../../src/rag/answer';
import { NOT_FOUND } from '../../src/rag/prompts';

const scores = (accuracy: number, completeness: number, relevance: number, reasoning = 'ok') =>
  JSON.stringify({ reasoning, accuracy, completeness, relevance });

const ok = (a: number, c: number, r: number): JudgeVerdict => ({
  status: 'ok',
  repairs: 0,
  scores: { reasoning: 'ok', accuracy: a, completeness: c, relevance: r },
});
const INVALID: JudgeVerdict = { status: 'invalid_output', repairs: 2, error: 'bad' };

const INPUT = {
  question: 'How do I read env vars?',
  referenceAnswer: 'Use Bun.env.',
  context: '[chunk:11] (runtime/env.md)\nRead variables with Bun.env.',
  answer: 'Use Bun.env.',
};

describe('judgeAnswer', () => {
  it('returns valid scores and sends temperature 0, the fixed seed and only the four fields', async () => {
    const chat = new FakeChatProvider([scores(5, 4, 3)]);
    let req: GenerateRequest | undefined;
    const spy: Pick<ChatProvider, 'judge'> = {
      judge: <T>(r: GenerateRequest, schema: z.ZodType<T>) => {
        req = r;
        return chat.judge(r, schema);
      },
    };
    expect(await judgeAnswer(spy, INPUT)).toEqual(ok(5, 4, 3));
    expect(req).toMatchObject({ temperature: JUDGE_TEMPERATURE, seed: JUDGE_SEED });
    expect(req?.messages).toEqual(buildJudgeMessages(INPUT));
    expect(req?.messages[1]?.content).toBe(
      `QUESTION: ${INPUT.question}\nREFERENCE ANSWER: ${INPUT.referenceAnswer}\n` +
        `CONTEXT: ${INPUT.context}\nASSISTANT ANSWER: ${INPUT.answer}`,
    );
  });

  it('accepts a score fixed in a repair round', async () => {
    const chat = new FakeChatProvider([scores(6, 4, 3), scores(5, 4, 3)]);
    expect(await judgeAnswer(chat, INPUT)).toEqual({ ...ok(5, 4, 3), repairs: 1 });
  });

  it('keeps a reply that fails validation after the repairs as invalid_output', async () => {
    const chat = new FakeChatProvider(['not json', '{"accuracy": 5}', scores(0, 4, 3)]);
    const v = await judgeAnswer(chat, INPUT);
    expect(v).toMatchObject({ status: 'invalid_output', repairs: 2 });
    expect(chat.calls).toHaveLength(3);
  });
});

const entry = (over: Partial<GoldenEntry> & Pick<GoldenEntry, 'id'>): GoldenEntry => ({
  split: 'dev',
  category: 'direct_fact',
  question: `question ${over.id}`,
  answerable: true,
  evidence: [{ source: 'guides/widget.md', span: `fact ${over.id}` }],
  keywords: ['widget'],
  reference_answer: `reference ${over.id}`,
  ...over,
});
const negative = (id: string, over: Partial<GoldenEntry> = {}) =>
  entry({ id, category: 'negative', answerable: false, evidence: [], keywords: [], ...over });

const result = (found: boolean | null, over: Partial<AnswerResult> = {}): AnswerResult => ({
  requestId: `req-${String(found)}`,
  found,
  rewrite: null,
  expand: null,
  answer: found === true ? 'an answer' : found === false ? NOT_FOUND : null,
  citations: found === true ? [{ chunkId: 1, sourcePath: 'guides/widget.md', headingPath: '' }] : [],
  droppedCitations: [],
  retrievedIds: [1, 2],
  context: '[chunk:1] (guides/widget.md)\nfact',
  status: found === null ? 'invalid_output' : 'ok',
  repairs: 0,
  latencyMs: 100,
  ...over,
});

/**
 * a1 human found=true valid citations; a2 llm found=true with a dropped citation; a3 abstains; a4 unusable output;
 * n1 negative abstains; n2 negative (llm) answers anyway.
 */
const ANSWERED: AnsweredEntry[] = [
  { entry: entry({ id: 'q001' }), result: result(true, { latencyMs: 400 }) },
  {
    entry: entry({ id: 'q002', origin: 'llm' }),
    result: result(true, { droppedCitations: [9], latencyMs: 300 }),
  },
  { entry: entry({ id: 'q003' }), result: result(false, { latencyMs: 200 }) },
  { entry: entry({ id: 'q004' }), result: result(null, { latencyMs: 600 }) },
  { entry: negative('q005'), result: result(false, { latencyMs: 100 }) },
  { entry: negative('q006', { origin: 'llm' }), result: result(true, { latencyMs: 500 }) },
];

/** A judge that replies per question (the order is shuffled), running the real repair loop. */
function routedJudge(replies: Record<string, string[]>) {
  const asked: string[] = [];
  const judge: Pick<ChatProvider, 'judge'> = {
    judge: (req, schema) =>
      generateObject(
        async () => {
          const question = /QUESTION: (.*)\n/.exec(String(req.messages[1]?.content))?.[1] ?? '';
          asked.push(question);
          const content = replies[question]?.shift();
          if (content === undefined) {
            throw new Error(`no scripted judge reply for ${question}`);
          }
          return { content, model: 'fake-judge', attempts: 1 };
        },
        req.messages,
        schema,
      ),
  };
  return { judge, asked };
}

const REPLIES = () => ({
  'question q001': [scores(5, 4, 3), scores(5, 4, 3)],
  'question q002': [scores(3, 2, 1), scores(3, 2, 2)],
});

describe('answerMetrics', () => {
  it('computes judge means, abstention and citation metrics by hand', async () => {
    const { judge } = routedJudge(REPLIES());
    const run = await judgeAll({ answered: ANSWERED, judge, seed: 7 });
    expect(answerMetrics(run.questions)).toEqual({
      nAnswerable: 4,
      nNegative: 2,
      scored: 2,
      accuracy: 4,
      completeness: 3,
      relevance: 2,
      // n1 abstains, n2 does not.
      abstentionAccuracy: 0.5,
      // a3 abstains; a4 (found=null) is no abstention.
      falseAbstentionRate: 0.25,
      // found=true: a1 valid, a2 dropped a citation, n2 valid.
      citationValidity: 2 / 3,
      invalidAnswerRate: 1 / 6,
    });
  });

  it('counts found=null as a failed abstention on a negative and keeps invalid judge output out of the means', () => {
    const base = { origin: 'human' as const, citationValid: null };
    expect(
      answerMetrics([
        { ...base, answerable: false, found: null, judge: null },
        { ...base, answerable: true, found: true, citationValid: true, judge: INVALID },
        {
          ...base,
          answerable: true,
          found: true,
          citationValid: true,
          judge: ok(2, 2, 2),
        },
      ]),
    ).toMatchObject({ abstentionAccuracy: 0, scored: 1, accuracy: 2, invalidAnswerRate: 1 / 3 });
  });

  it('returns null rates for empty groups', () => {
    expect(answerMetrics([])).toEqual({
      nAnswerable: 0,
      nNegative: 0,
      scored: 0,
      accuracy: null,
      completeness: null,
      relevance: null,
      abstentionAccuracy: null,
      falseAbstentionRate: null,
      citationValidity: null,
      invalidAnswerRate: null,
    });
  });
});

describe('judgeAll', () => {
  it('judges only answerable found=true entries, in a reproducible shuffled order, and repeats a sample', async () => {
    const a = routedJudge(REPLIES());
    const run = await judgeAll({ answered: ANSWERED, judge: a.judge, seed: 7 });
    const b = routedJudge(REPLIES());
    await judgeAll({ answered: ANSWERED, judge: b.judge, seed: 7 });

    expect(a.asked).toEqual(b.asked);
    // Two first verdicts, then the repeat of the first answer in the shuffled order.
    expect(a.asked).toHaveLength(3);
    expect(a.asked[2]).toBe(a.asked[0]);
    expect(run.shuffleSeed).toBe(7);
    expect(run.questions.map((q) => q.id)).toEqual(['q001', 'q002', 'q003', 'q004', 'q005', 'q006']);
    expect(run.questions.map((q) => q.judge !== null)).toEqual([true, true, false, false, false, false]);
    expect(run.questions.filter((q) => q.judgeRepeat !== null)).toHaveLength(1);
  });

  it('counts a judge reply that fails validation instead of dropping it', async () => {
    const { judge } = routedJudge({
      'question q001': [scores(5, 4, 3), scores(5, 4, 3)],
      'question q002': ['nope', 'nope', 'nope', 'nope', 'nope', 'nope'],
    });
    const run = await judgeAll({ answered: ANSWERED, judge, seed: 7 });
    const q2 = run.questions.find((q) => q.id === 'q002');
    expect(q2?.judge).toMatchObject({ status: 'invalid_output', repairs: 2 });
    expect(answerMetrics(run.questions)).toMatchObject({ scored: 1, accuracy: 5 });
  });
});

describe('answerAll', () => {
  it('answers the split sorted by id', async () => {
    const asked: string[] = [];
    const out = await answerAll({
      entries: [entry({ id: 'q002' }), entry({ id: 'q001' }), entry({ id: 'q003', split: 'test' })],
      split: 'dev',
      answer: async (e) => {
        asked.push(e.id);
        return result(true);
      },
    });
    expect(asked).toEqual(['q001', 'q002']);
    expect(out.map((a) => a.entry.id)).toEqual(['q001', 'q002']);
  });
});

describe('self-consistency', () => {
  it('shares identical scores over valid pairs, overall and per dimension', () => {
    expect(
      selfConsistency([
        { first: ok(5, 4, 3), repeat: ok(5, 4, 3) },
        { first: ok(5, 4, 3), repeat: ok(5, 3, 3) },
        { first: ok(5, 4, 3), repeat: INVALID },
      ]),
    ).toEqual({
      sampled: 3,
      repeatInvalid: 1,
      pairs: 2,
      identicalShare: 0.5,
      perDimension: { accuracy: 1, completeness: 0.5, relevance: 1 },
    });
  });

  it('draws 10 percent rounded up, at least one', () => {
    expect([0, 1, 10, 13, 30].map(sampleSize)).toEqual([0, 1, 1, 2, 3]);
  });

  it('shuffles reproducibly with the seed', () => {
    const items = Array.from({ length: 10 }, (_, i) => i);
    const a = shuffle(items, 123);
    expect(shuffle(items, 123)).toEqual(a);
    expect([...a].sort((x, y) => x - y)).toEqual(items);
    expect(shuffle(items, 124)).not.toEqual(a);
    expect(items).toEqual(Array.from({ length: 10 }, (_, i) => i));
  });
});

const CONFIG: RunConfig = {
  chunkConfig: 'md-512-64',
  embed: 'nomic-embed-text',
  k: 5,
  search: 'vector',
  split: 'dev',
  rewrite: 'off',
  expand: 'off',
};
const fake = (name: string) => ({ provider: 'fake', name, tag: null, digest: null });
const META = {
  experimentId: 'baseline',
  runId: 'eval-answers-test',
  startedAt: new Date('2026-11-15T14:03:12.345Z'),
  finishedAt: new Date('2026-11-15T14:04:00.000Z'),
  gitSha: 'a'.repeat(40),
  gitDirty: false,
  datasetSha256: 'b'.repeat(64),
  corpusSha: 'c'.repeat(40),
  config: CONFIG,
  variedKey: null,
  models: { embed: fake('nomic-embed-text'), chat: fake('chat-model'), judge: fake('judge-model') },
};
const KPIS = {
  latencyMs: { p50: 350, p95: 587.123456 },
  calls: [
    {
      operation: 'judge',
      model: 'judge-model',
      calls: 3,
      avgInputTokens: 123.456789,
      avgOutputTokens: 40,
      estimatedTokenShare: 0,
      retryRate: 1 / 3,
      invalidOutputRate: 0,
      costUsd: 0,
      latencyP50Ms: 812.5,
      latencyP95Ms: 1234.56789,
    },
  ],
};

describe('buildAnswerResult', () => {
  async function build() {
    const { judge } = routedJudge(REPLIES());
    return buildAnswerResult(META, await judgeAll({ answered: ANSWERED, judge, seed: 7 }), KPIS);
  }

  it('writes keys in a fixed order, rounds, and reports all, human and llm', async () => {
    const out = await build();
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
      'judge',
      'shuffleSeed',
      'counts',
      'aggregate',
      'selfConsistency',
      'kpis',
      'questions',
    ]);
    expect(out.counts).toEqual({
      questions: 6,
      answerable: 4,
      negative: 2,
      judged: 2,
      judgeInvalid: 0,
      invalidAnswers: 1,
      rewriteFallbacks: null,
      expandFallbacks: null,
    });
    expect(out.aggregate.all).toMatchObject({ citationValidity: 0.6667, invalidAnswerRate: 0.1667 });
    expect(out.aggregate.human).toEqual({
      nAnswerable: 3,
      nNegative: 1,
      scored: 1,
      accuracy: 5,
      completeness: 4,
      relevance: 3,
      abstentionAccuracy: 1,
      falseAbstentionRate: 0.3333,
      citationValidity: 1,
      invalidAnswerRate: 0.25,
    });
    expect(out.aggregate.llm).toMatchObject({
      scored: 1,
      accuracy: 3,
      abstentionAccuracy: 0,
      citationValidity: 0.5,
    });
    expect(out.selfConsistency).toMatchObject({ sampled: 1, pairs: 1 });
    expect(out.kpis.latencyMs).toEqual({ p50: 350, p95: 587.1235 });
    expect(out.kpis.calls[0]).toMatchObject({ latencyP50Ms: 812.5, latencyP95Ms: 1234.5679 });
    expect(out.kpis.calls[0]).toMatchObject({ avgInputTokens: 123.4568, retryRate: 0.3333 });
    expect(out.judge).toEqual({ temperature: 0, seed: 42 });
    expect(Object.keys(out.questions[0] ?? {})).toEqual([
      'id',
      'category',
      'origin',
      'answerable',
      'requestId',
      'found',
      'status',
      'repairs',
      'latencyMs',
      'answer',
      'citations',
      'droppedCitations',
      'retrievedIds',
      'rewrite',
      'expand',
      'citationValid',
      'judge',
      'judgeRepeat',
    ]);
    expect(out.questions[0]?.judge).toEqual({
      status: 'ok',
      repairs: 0,
      reasoning: 'ok',
      accuracy: 5,
      completeness: 4,
      relevance: 3,
    });
    expect(out.questions[2]?.judge).toBeNull();
  });

  it('prints a summary with the seed and one row per origin group', async () => {
    const lines = formatAnswerSummary(await build()).split('\n');
    expect(lines[0]).toContain('chat chat-model, judge judge-model');
    expect(lines[0]).toContain('shuffle seed 7');
    expect(lines.slice(2, 5).map((l) => l.split(/\s+/)[0])).toEqual(['all', 'human', 'llm']);
    expect(lines[5]).toMatch(/^self-consistency: 1 sampled, 1 pairs/);
    expect(lines[6]).toBe('latency p50 350 ms, p95 587.1235 ms');
  });
});
