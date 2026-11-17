import { describe, expect, it } from 'vitest';

import { chunkHash, documentHash, embedText, sha256 } from '../../src/ingest/hash';
import { parseMarkdown } from '../../src/ingest/loadMarkdown';

describe('sha256', () => {
  it('matches known hex digests', () => {
    expect(sha256('')).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    expect(sha256('abc')).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  });
});

describe('documentHash', () => {
  const raw = (description: string, title = 'Widget') =>
    `---\ntitle: ${title}\ndescription: ${description}\n---\n\nWidget is a fictional tool.\n`;

  it('changes with the description (M19b), but not with other frontmatter keys', () => {
    expect(documentHash(parseMarkdown(raw('one'), 'x'))).not.toBe(
      documentHash(parseMarkdown(raw('two'), 'x')),
    );
    const withKey = raw('one').replace('---\n\n', 'mode: center\n---\n\n');
    expect(documentHash(parseMarkdown(withKey, 'x'))).toBe(documentHash(parseMarkdown(raw('one'), 'x')));
  });

  it('keeps the pre-M19b hash without a description', () => {
    const doc = { title: 'Widget', body: 'text' };
    expect(documentHash(doc)).toBe(sha256('Widget\0text'));
    expect(documentHash(parseMarkdown(raw(''), 'x'))).toBe(
      documentHash(parseMarkdown('---\ntitle: Widget\n---\n\nWidget is a fictional tool.\n', 'x')),
    );
    expect(documentHash({ ...doc, description: 'text' })).not.toBe(documentHash(doc));
  });

  it('changes with the title or the body', () => {
    const base = documentHash({ title: 'Widget', body: 'text' });
    expect(documentHash({ title: 'Gadget', body: 'text' })).not.toBe(base);
    expect(documentHash({ title: 'Widget', body: 'text!' })).not.toBe(base);
  });

  it('keeps title and body apart', () => {
    expect(documentHash({ title: 'ab', body: 'c' })).not.toBe(documentHash({ title: 'a', body: 'bc' }));
  });
});

describe('embedText and chunkHash', () => {
  it('prefixes the heading path as a context line', () => {
    expect(embedText({ headingPath: 'Guide > Install', content: 'Run it.' })).toBe(
      'Guide > Install\n\nRun it.',
    );
    expect(embedText({ headingPath: '', content: 'Run it.' })).toBe('Run it.');
  });

  it('hashes the embed text, so the heading path is part of the chunk hash', () => {
    const chunk = { headingPath: 'Guide > Install', content: 'Run it.' };
    expect(chunkHash(chunk)).toBe(sha256('Guide > Install\n\nRun it.'));
    expect(chunkHash({ ...chunk, headingPath: 'Guide > Build' })).not.toBe(chunkHash(chunk));
  });
});
