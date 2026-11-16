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
