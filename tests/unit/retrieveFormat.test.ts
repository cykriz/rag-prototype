import { Document } from '@langchain/core/documents';
import { describe, expect, it } from 'vitest';

import { formatHits, preview } from '../../src/cli/retrieve';
import type { ChunkMetadata } from '../../src/retrieval/pgRetriever';

const doc = (chunk_id: number, score: number, heading_path: string, pageContent: string) =>
  new Document<ChunkMetadata>({
    pageContent,
    metadata: { chunk_id, source_path: 'runtime/env.md', heading_path, score },
  });

describe('preview', () => {
  it('collapses whitespace and keeps short text as is', () => {
    expect(preview('  Bun reads\n\n.env   files ')).toBe('Bun reads .env files');
  });

  it('cuts long text to the limit with an ellipsis', () => {
    const out = preview('word '.repeat(50), 20);
    expect(out).toHaveLength(20);
    expect(out.endsWith('…')).toBe(true);
  });
});

describe('formatHits', () => {
  const meta = { chunkConfig: 'md-512-64', model: 'nomic-embed-text', k: 2, search: 'vector' as const };

  it('prints rank, score, chunk ID, source with heading path and a preview line', () => {
    const out = formatHits('read env', meta, [
      doc(12, 0.71345, 'Environment variables', 'Bun reads .env files.'),
      doc(7, 0.7, '', 'Second chunk.'),
    ]);
    expect(out.split('\n')).toEqual([
      'Top 2 for "read env" (md-512-64 / nomic-embed-text, vector)',
      ' 1  0.7135  #12  runtime/env.md > Environment variables',
      '    Bun reads .env files.',
      ' 2  0.7000  #7  runtime/env.md',
      '    Second chunk.',
    ]);
  });

  it('says so when the config holds fewer chunks than k', () => {
    const out = formatHits('read env', meta, [doc(1, 0.1, '', 'Only one.')]);
    expect(out.split('\n').at(-1)).toBe('only 1 chunks in md-512-64');
  });

  it('names the search mode and says "matching" when fts finds fewer than k', () => {
    const out = formatHits('read env', { ...meta, search: 'fts' }, [doc(1, 0.1, '', 'Only one.')]);
    expect(out.split('\n')[0]).toBe('Top 2 for "read env" (md-512-64 / nomic-embed-text, fts)');
    expect(out.split('\n').at(-1)).toBe('only 1 matching chunks in md-512-64');
  });
});
