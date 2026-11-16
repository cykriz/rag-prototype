import { describe, expect, it } from 'vitest';

import { parseQuestionArgs } from '../../src/cli/common';

describe('parseQuestionArgs --search', () => {
  it('defaults to vector', () => {
    expect(parseQuestionArgs(['how', 'to', 'test'])).toMatchObject({
      question: 'how to test',
      pipeline: { search: 'vector', rewrite: 'off' },
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
