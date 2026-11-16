import { describe, expect, it } from 'vitest';

import { parseQuestionArgs } from '../../src/cli/common';

describe('parseQuestionArgs --search', () => {
  it('defaults to vector', () => {
    expect(parseQuestionArgs(['how', 'to', 'test'])).toMatchObject({
      question: 'how to test',
      search: 'vector',
    });
  });

  it('takes fts and hybrid', () => {
    expect(parseQuestionArgs(['q', '--search', 'fts'])).toMatchObject({ search: 'fts' });
    expect(parseQuestionArgs(['q', '--search', 'hybrid'])).toMatchObject({ search: 'hybrid' });
  });

  it('rejects an unknown mode', () => {
    expect(() => parseQuestionArgs(['q', '--search', 'bm25'])).toThrow(
      '--search must be one of vector, fts, hybrid',
    );
  });
});
