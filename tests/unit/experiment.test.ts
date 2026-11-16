import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseEvalArgs, requiredModels } from '../../src/cli/eval';
import { parseExperimentArgs, planRuns } from '../../src/cli/experiment';
import { loadConfig } from '../../src/config';
import { ExperimentFile, expandExperiment, loadExperiment, singleConfig } from '../../src/eval/experiment';

const BASELINE = join(import.meta.dirname, '../../experiments/baseline.json');

const parse = (raw: unknown) => ExperimentFile.parse(raw);
const BASE = { chunkConfig: 'md-512-64', embed: 'nomic-embed-text', k: 5, search: 'vector', split: 'test' };
/** BASE as a parsed RunConfig: rewrite defaults to off. */
const RUN = { ...BASE, rewrite: 'off' };

describe('experiment files', () => {
  it('experiments/baseline.json is one retrieval config on the test split', async () => {
    const exp = await loadExperiment(BASELINE);
    expect(exp.id).toBe('baseline');
    expect(singleConfig(exp, 'retrieval')).toEqual(RUN);
  });

  it('experiments/exp-rewrite.json varies rewrite off, on at the M14 config', async () => {
    const exp = await loadExperiment(join(import.meta.dirname, '../../experiments/exp-rewrite.json'));
    expect(expandExperiment(exp)).toEqual([
      { run: { ...RUN, chunkConfig: 'md-1024-128', rewrite: 'off' }, variedKey: 'rewrite', label: 'off' },
      { run: { ...RUN, chunkConfig: 'md-1024-128', rewrite: 'on' }, variedKey: 'rewrite', label: 'on' },
    ]);
    expect(exp.evals).toEqual(['retrieval', 'answers']);
  });

  it('accepts the §7 matrix format, but singleConfig refuses a vary', () => {
    const exp = parse({
      id: 'exp-chunk-size',
      base: { embed: 'nomic-embed-text', k: 5, search: 'vector', split: 'test' },
      vary: { chunkConfig: ['md-256-32', 'md-512-64', 'md-1024-128'] },
      evals: ['retrieval'],
    });
    expect(() => singleConfig(exp, 'retrieval')).toThrow(/run the matrix with bun run experiment <file>/);
  });

  it('singleConfig names a missing base field and a missing eval type', () => {
    const { chunkConfig: _, ...noChunk } = BASE;
    expect(() => singleConfig(parse({ id: 'x', base: noChunk, evals: ['retrieval'] }), 'retrieval')).toThrow(
      /base\.chunkConfig/,
    );
    expect(() => singleConfig(parse({ id: 'x', base: BASE, evals: ['answers'] }), 'retrieval')).toThrow(
      /does not include "retrieval"/,
    );
  });

  it('rejects unknown values', () => {
    const bad = (base: Record<string, unknown>) =>
      ExperimentFile.safeParse({ id: 'x', base: { ...BASE, ...base }, evals: ['retrieval'] }).success;
    expect(bad({ search: 'bm25' })).toBe(false);
    expect(bad({ search: 'hybrid' })).toBe(true);
    expect(bad({ embed: 'unknown-model' })).toBe(false);
    expect(bad({ chunkConfig: '512' })).toBe(false);
    expect(bad({ k: 0 })).toBe(false);
    expect(bad({ split: 'all' })).toBe(false);
    expect(bad({})).toBe(true);
  });
});

describe('parseEvalArgs', () => {
  it('parses the retrieval command', () => {
    expect(parseEvalArgs(['retrieval', '--config', 'experiments/baseline.json', '--split', 'dev'])).toEqual({
      command: 'retrieval',
      config: 'experiments/baseline.json',
      split: 'dev',
      dataset: 'eval/golden.jsonl',
      out: 'results',
      seed: undefined,
    });
    expect(parseEvalArgs(['retrieval', '--help'])).toBeUndefined();
  });

  it('parses the answers command with a seed', () => {
    expect(parseEvalArgs(['answers', '--config', 'x.json', '--seed', '4294967295'])).toMatchObject({
      command: 'answers',
      seed: 4294967295,
    });
    expect(parseEvalArgs(['answers', '--config', 'x.json'])?.seed).toBeUndefined();
  });

  it('throws on usage errors', () => {
    expect(() => parseEvalArgs(['retrieval'])).toThrow(/--config/);
    expect(() => parseEvalArgs(['retrieval', '--config', 'x.json', '--split', 'all'])).toThrow(/--split/);
    expect(() => parseEvalArgs(['report', '--config', 'x.json'])).toThrow(/Unknown eval command/);
    expect(() => parseEvalArgs(['retrieval', '--config', 'x.json', '--seed', '1'])).toThrow(/answers/);
    expect(() => parseEvalArgs(['answers', '--config', 'x.json', '--seed', '-1'])).toThrow(/--seed/);
    expect(() => parseEvalArgs(['answers', '--config', 'x.json', '--seed', '4294967296'])).toThrow(/--seed/);
    expect(() => parseEvalArgs(['retrieval', '--config', 'x.json', '--bogus'])).toThrow();
  });
});

describe('expandExperiment', () => {
  const { chunkConfig: _, ...noChunk } = BASE;
  const chunkSize = parse({
    id: 'exp-chunk-size',
    base: noChunk,
    vary: { chunkConfig: ['md-256-32', 'md-512-64', 'md-1024-128'] },
    evals: ['retrieval', 'answers'],
  });

  it('gives one variant per value in file order, labelled with the value', () => {
    expect(expandExperiment(chunkSize)).toEqual([
      { run: { ...RUN, chunkConfig: 'md-256-32' }, variedKey: 'chunkConfig', label: 'md-256-32' },
      { run: { ...RUN, chunkConfig: 'md-512-64' }, variedKey: 'chunkConfig', label: 'md-512-64' },
      { run: { ...RUN, chunkConfig: 'md-1024-128' }, variedKey: 'chunkConfig', label: 'md-1024-128' },
    ]);
    const { k: __, ...noK } = BASE;
    const labels = expandExperiment(
      parse({ id: 'exp-k', base: noK, vary: { k: [3, 10] }, evals: ['retrieval'] }),
    );
    expect(labels.map((v) => v.label)).toEqual(['3', '10']);
  });

  it('gives the single base config without vary', () => {
    expect(expandExperiment(parse({ id: 'baseline', base: BASE, evals: ['retrieval'] }))).toEqual([
      { run: RUN, variedKey: null, label: null },
    ]);
  });

  it('takes rewrite in base and refuses it in base and vary at once', () => {
    const on = parse({ id: 'x', base: { ...BASE, rewrite: 'on' }, evals: ['retrieval'] });
    expect(expandExperiment(on)[0]?.run.rewrite).toBe('on');
    // The base schema has no default: an unset base.rewrite stays undefined, so varying it is allowed.
    expect(parse({ id: 'x', base: BASE, evals: ['retrieval'] }).base.rewrite).toBeUndefined();
    const both = ExperimentFile.safeParse({
      id: 'x',
      base: { ...BASE, rewrite: 'off' },
      vary: { rewrite: ['off', 'on'] },
      evals: ['retrieval'],
    });
    expect(both.error?.message).toMatch(/also varied/);
    expect(
      ExperimentFile.safeParse({ id: 'x', base: { ...BASE, rewrite: 'yes' }, evals: ['retrieval'] }).success,
    ).toBe(false);
  });

  it('names a base field that is neither set nor varied', () => {
    const { embed: __, ...noEmbed } = noChunk;
    expect(() =>
      expandExperiment(
        parse({ id: 'x', base: noEmbed, vary: { chunkConfig: ['md-256-32'] }, evals: ['retrieval'] }),
      ),
    ).toThrow(/x: base\.embed/);
  });

  it('allows one varied key only, unique values, not also set in base', () => {
    const ok = (raw: Record<string, unknown>) =>
      ExperimentFile.safeParse({ id: 'x', evals: ['retrieval'], ...raw });
    expect(ok({ base: noChunk, vary: { chunkConfig: ['md-256-32'] } }).success).toBe(true);
    expect(ok({ base: {}, vary: { chunkConfig: ['md-256-32'], k: [3] } }).error?.message).toMatch(
      /one variable/,
    );
    expect(ok({ base: noChunk, vary: { chunkConfig: ['md-256-32', 'md-256-32'] } }).error?.message).toMatch(
      /unique/,
    );
    expect(ok({ base: BASE, vary: { chunkConfig: ['md-256-32'] } }).error?.message).toMatch(/also varied/);
  });

  it('plans all retrieval runs before the answers runs', () => {
    const plan = planRuns(chunkSize, undefined).map((r) => `${r.evalType} ${r.variant.label}`);
    expect(plan).toEqual([
      'retrieval md-256-32',
      'retrieval md-512-64',
      'retrieval md-1024-128',
      'answers md-256-32',
      'answers md-512-64',
      'answers md-1024-128',
    ]);
  });

  it('applies --split to every variant and refuses it together with vary.split', () => {
    expect(planRuns(chunkSize, 'dev').every((r) => r.variant.run.split === 'dev')).toBe(true);
    const { split: __, ...noSplit } = BASE;
    const splits = parse({ id: 'x', base: noSplit, vary: { split: ['dev', 'test'] }, evals: ['retrieval'] });
    expect(() => planRuns(splits, 'dev')).toThrow(/--split cannot override vary\.split/);
  });
});

describe('parseExperimentArgs', () => {
  it('parses the file and flags', () => {
    expect(parseExperimentArgs(['experiments/exp-k.json', '--split', 'dev', '--seed', '7'])).toEqual({
      file: 'experiments/exp-k.json',
      split: 'dev',
      seed: 7,
      dataset: 'eval/golden.jsonl',
      out: 'results',
    });
    expect(parseExperimentArgs(['--help'])).toBeUndefined();
  });

  it('throws on usage errors', () => {
    expect(() => parseExperimentArgs([])).toThrow(/one experiment file/);
    expect(() => parseExperimentArgs(['a.json', 'b.json'])).toThrow(/one experiment file/);
    expect(() => parseExperimentArgs(['a.json', '--split', 'all'])).toThrow(/--split/);
    expect(() => parseExperimentArgs(['a.json', '--seed', 'x'])).toThrow(/--seed/);
  });
});

describe('requiredModels', () => {
  const config = loadConfig({ CHAT_PROVIDER: 'fake', EMBED_PROVIDER: 'fake' });
  const roles = (evalType: 'retrieval' | 'answers', rewrite: 'off' | 'on') =>
    requiredModels(config, evalType, { rewrite }).map(([role]) => role);

  it('adds the chat model to a retrieval run only with rewrite on', () => {
    expect(roles('retrieval', 'off')).toEqual(['embed']);
    expect(roles('retrieval', 'on')).toEqual(['embed', 'chat']);
    expect(roles('answers', 'on')).toEqual(['embed', 'chat', 'judge']);
  });
});
