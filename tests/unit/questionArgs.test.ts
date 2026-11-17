import { describe, expect, it } from 'vitest';

import { parseQuestionArgs } from '../../src/cli/common';

describe('parseQuestionArgs --search', () => {
  it('defaults to vector', () => {
    expect(parseQuestionArgs(['how', 'to', 'test'])).toMatchObject({
      question: 'how to test',
      pipeline: { search: 'vector', rewrite: 'off', expand: 'off' },
    });
  });

  it('takes fts and hybrid', () => {
    expect(parseQuestionArgs(['q', '--search', 'fts'])).toMatchObject({ pipeline: { search: 'fts' } });
    expect(parseQuestionArgs(['q', '--search', 'hybrid'])).toMatchObject({ pipeline: { search: 'hybrid' } });
  });

  it('rejects an unknown mode', () => {
    expect(() => parseQuestionArgs(['q', '--search', 'bm25'])).toThrow(
      '--search must be one of vector, fts, hybrid',
    );
  });
});

describe('parseQuestionArgs --rewrite', () => {
  it('is off by default and on with the flag, next to --search', () => {
    expect(parseQuestionArgs(['q'])).toMatchObject({ pipeline: { rewrite: 'off' } });
    expect(parseQuestionArgs(['q', '--rewrite', '--search', 'hybrid'])).toMatchObject({
      pipeline: { search: 'hybrid', rewrite: 'on' },
    });
  });
});

describe('parseQuestionArgs --expand', () => {
  it('is off by default and on with the flag, next to --search', () => {
    expect(parseQuestionArgs(['q'])).toMatchObject({ pipeline: { expand: 'off' } });
    expect(parseQuestionArgs(['q', '--expand', '--search', 'fts'])).toMatchObject({
      pipeline: { search: 'fts', rewrite: 'off', expand: 'on' },
    });
  });

  it('refuses --rewrite together with --expand', () => {
    expect(() => parseQuestionArgs(['q', '--rewrite', '--expand'])).toThrow(
      'rewrite and expand cannot both be on',
    );
  });
});

describe('parseQuestionArgs --rerank', () => {
  it('is off by default and combines with --rewrite or --expand', () => {
    expect(parseQuestionArgs(['q'])).toMatchObject({ pipeline: { rerank: 'off' } });
    expect(parseQuestionArgs(['q', '--rerank', '--expand'])).toMatchObject({
      pipeline: { search: 'vector', rewrite: 'off', expand: 'on', rerank: 'on' },
    });
    expect(parseQuestionArgs(['q', '--rewrite', '--rerank'])).toMatchObject({
      pipeline: { rewrite: 'on', rerank: 'on' },
    });
  });
});

describe('parseQuestionArgs --pipeline', () => {
  const ask = { askFlags: true };

  it('is only accepted when the caller enables it (ask, not retrieve)', () => {
    expect(() => parseQuestionArgs(['q', '--pipeline', 'agent'])).toThrow();
    expect(parseQuestionArgs(['q'])).toMatchObject({ agent: false });
  });

  it('defaults to baseline and takes agent next to --search', () => {
    expect(parseQuestionArgs(['q'], ask)).toMatchObject({ agent: false });
    expect(parseQuestionArgs(['q', '--pipeline', 'agent', '--search', 'hybrid'], ask)).toMatchObject({
      agent: true,
      pipeline: { search: 'hybrid', rewrite: 'off', expand: 'off', rerank: 'off' },
    });
  });

  it('rejects unknown values, the query stages and a k above the tool limit with agent', () => {
    expect(() => parseQuestionArgs(['q', '--pipeline', 'graph'], ask)).toThrow('--pipeline must be one of');
    for (const flag of ['--rewrite', '--expand', '--rerank']) {
      expect(() => parseQuestionArgs(['q', '--pipeline', 'agent', flag], ask)).toThrow('does not combine');
    }
    expect(() => parseQuestionArgs(['q', '--pipeline', 'agent', '--k', '11'], ask)).toThrow('at most 10');
    expect(parseQuestionArgs(['q', '--rerank', '--k', '11'], ask)).toMatchObject({ agent: false, k: 11 });
  });
});

describe('parseQuestionArgs --style (M19c)', () => {
  const ask = { askFlags: true };

  it('is only accepted when the caller enables it (ask, not retrieve)', () => {
    expect(() => parseQuestionArgs(['q', '--style', 'complete'])).toThrow();
    expect(parseQuestionArgs(['q'])).toMatchObject({ answerStyle: 'concise' });
  });

  it('defaults to complete for ask, concise for the agent, and takes either explicitly', () => {
    expect(parseQuestionArgs(['q'], ask)).toMatchObject({ answerStyle: 'complete' });
    expect(parseQuestionArgs(['q', '--pipeline', 'agent'], ask)).toMatchObject({
      agent: true,
      answerStyle: 'concise',
    });
    expect(parseQuestionArgs(['q', '--style', 'concise'], ask)).toMatchObject({ answerStyle: 'concise' });
    expect(parseQuestionArgs(['q', '--style', 'complete', '--rerank'], ask)).toMatchObject({
      answerStyle: 'complete',
      agent: false,
      pipeline: { rerank: 'on' },
    });
  });

  it('rejects unknown values and complete with the agent', () => {
    expect(() => parseQuestionArgs(['q', '--style', 'long'], ask)).toThrow('--style must be one of');
    expect(() => parseQuestionArgs(['q', '--pipeline', 'agent', '--style', 'complete'], ask)).toThrow(
      'does not combine with answer style complete',
    );
    expect(parseQuestionArgs(['q', '--pipeline', 'agent', '--style', 'concise'], ask)).toMatchObject({
      agent: true,
      answerStyle: 'concise',
    });
  });
});
