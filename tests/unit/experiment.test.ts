import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseEvalArgs } from '../../src/cli/eval';
import { ExperimentFile, loadExperiment, singleConfig } from '../../src/eval/experiment';

const BASELINE = join(import.meta.dirname, '../../experiments/baseline.json');

const parse = (raw: unknown) => ExperimentFile.parse(raw);
const BASE = { chunkConfig: 'md-512-64', embed: 'nomic-embed-text', k: 5, search: 'vector', split: 'test' };

describe('experiment files', () => {
  it('experiments/baseline.json is one retrieval config on the test split', async () => {
    const exp = await loadExperiment(BASELINE);
    expect(exp.id).toBe('baseline');
    expect(singleConfig(exp, 'retrieval')).toEqual(BASE);
  });

  it('accepts the §7 matrix format, but singleConfig refuses a vary', () => {
    const exp = parse({
      id: 'exp-chunk-size',
      base: { embed: 'nomic-embed-text', k: 5, search: 'vector', split: 'test' },
      vary: { chunkConfig: ['md-256-32', 'md-512-64', 'md-1024-128'] },
      evals: ['retrieval'],
    });
    expect(() => singleConfig(exp, 'retrieval')).toThrow(/bun run experiment \(M13a\)/);
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
    expect(bad({ search: 'hybrid' })).toBe(false);
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
    });
    expect(parseEvalArgs(['retrieval', '--help'])).toBeUndefined();
  });

  it('throws on usage errors', () => {
    expect(() => parseEvalArgs(['retrieval'])).toThrow(/--config/);
    expect(() => parseEvalArgs(['retrieval', '--config', 'x.json', '--split', 'all'])).toThrow(/--split/);
    expect(() => parseEvalArgs(['answers', '--config', 'x.json'])).toThrow(/Unknown eval command/);
    expect(() => parseEvalArgs(['retrieval', '--config', 'x.json', '--bogus'])).toThrow();
  });
});
