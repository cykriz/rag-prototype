import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { beforeAll, describe, expect, it } from 'vitest';

import { chunkDocument, chunkMarkdown, parseChunkConfigKey, plainChunkKey } from '../../src/ingest/chunk';
import { findFences, findHeadings } from '../../src/ingest/fences';
import { parseMarkdown } from '../../src/ingest/loadMarkdown';
import { countTokens } from '../../src/llm/tokens';

const md64 = parseChunkConfigKey('md-64-8');

describe('parseChunkConfigKey', () => {
  it('parses a valid key', () => {
    expect(parseChunkConfigKey('md-512-64')).toEqual({
      key: 'md-512-64',
      chunkSize: 512,
      chunkOverlap: 64,
      splitter: 'markdown',
    });
  });

  it('parses an mdfm key as the frontmatter splitter (M19b)', () => {
    expect(parseChunkConfigKey('mdfm-1024-128')).toEqual({
      key: 'mdfm-1024-128',
      chunkSize: 1024,
      chunkOverlap: 128,
      splitter: 'markdown-frontmatter',
    });
  });

  it.each([
    'md-0-0',
    'txt-512-64',
    'md-512',
    'md-512-64-1',
    'fm-1024-128',
    'mdfm-1024',
    'mdfm-0-0',
    'MDFM-64-8',
  ])('rejects %s', (key) => {
    expect(() => parseChunkConfigKey(key)).toThrow(/expected md-<size>-<overlap> or mdfm-<size>-<overlap>/);
  });

  it('maps an mdfm key to its md key, anything else to null', () => {
    expect(plainChunkKey('mdfm-1024-128')).toBe('md-1024-128');
    expect(plainChunkKey('md-1024-128')).toBeNull();
    expect(plainChunkKey('mdfm-1024')).toBeNull();
  });

  it('leaves overlap >= size to the splitter', async () => {
    await expect(chunkMarkdown('Some text.', 'T', parseChunkConfigKey('md-128-128'))).rejects.toThrow(
      /chunkOverlap/,
    );
  });
});

describe('findFences / findHeadings', () => {
  it('handles indented, tilde, longer closing and unclosed fences', () => {
    const text = '  ```ts\n  a\n  ```\n~~~\n```\n~~~~\n````\nb\n```\nc\n````\n```\nopen';
    const ranges = findFences(text).map((r) => text.slice(r.start, r.end));
    expect(ranges).toEqual(['  ```ts\n  a\n  ```', '~~~\n```\n~~~~', '````\nb\n```\nc\n````', '```\nopen']);
  });

  it('skips headings inside fences and strips closing hashes', () => {
    const text = '# Title #\n```sh\n# comment\n```\n  ## Sub';
    expect(findHeadings(text)).toEqual([
      { offset: 0, level: 1, text: 'Title' },
      { offset: text.indexOf('## Sub'), level: 2, text: 'Sub' },
    ]);
  });
});

describe('chunkMarkdown on the fixture', () => {
  let title: string;
  let body: string;
  let fence: string;

  beforeAll(async () => {
    const raw = await readFile(join(import.meta.dirname, '../fixtures/docs/widget-guide.mdx'), 'utf8');
    ({ title, body } = parseMarkdown(raw, 'widget-guide'));
    fence = body.slice(body.indexOf('```ts'), body.indexOf('};\n```') + 5);
  });

  it('produces the expected chunks and heading paths', async () => {
    const chunks = await chunkMarkdown(body, title, md64);
    expect(chunks.map((c) => c.ordinal)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(chunks.map((c) => c.headingPath)).toEqual([
      'Widget Guide',
      'Widget Guide > Install',
      'Widget Guide > Install > From source',
      'Widget Guide > Configure',
      'Widget Guide > Configure',
      'Widget Guide > Configure',
      'Widget Guide > Troubleshoot',
    ]);
    expect(chunks.some((c) => c.headingPath.includes('not a heading'))).toBe(false);
  });

  it.each([64, 128, 256])('keeps every chunk within %i tokens', async (size) => {
    const chunks = await chunkMarkdown(body, title, parseChunkConfigKey(`md-${size}-${size / 8}`));
    for (const c of chunks) {
      expect(c.tokenCount).toBe(countTokens(c.content));
      expect(c.tokenCount).toBeLessThanOrEqual(size);
    }
  });

  it('keeps a fitting fence whole: every chunk touching it contains all of it', async () => {
    const chunks = await chunkMarkdown(body, title, md64);
    const fenceLines = fence.split('\n').filter((l) => l.trim().length > 3);
    const touching = chunks.filter((c) => fenceLines.some((l) => c.content.includes(l)));
    expect(touching.length).toBeGreaterThan(0);
    for (const c of touching) {
      expect(c.content).toContain(fence);
    }
  });

  it('negative control: the plain markdown splitter cuts this fence', async () => {
    const plain = RecursiveCharacterTextSplitter.fromLanguage('markdown', {
      chunkSize: 64,
      chunkOverlap: 8,
      lengthFunction: countTokens,
    });
    const chunks = await plain.splitText(body);
    expect(chunks.some((c) => c.includes(fence))).toBe(false);
  });

  it('is deterministic', async () => {
    expect(await chunkMarkdown(body, title, md64)).toEqual(await chunkMarkdown(body, title, md64));
  });
});

describe('chunkMarkdown edge cases', () => {
  // A one-line fence whose token count grows by one per extra " a".
  const fenceWith = (target: number): string => {
    for (let k = 1; k < 400; k += 1) {
      const f = `\`\`\`txt\nx${' a'.repeat(k)}\n\`\`\``;
      if (countTokens(`\n\n${f}`) === target) {
        return f;
      }
    }
    throw new Error(`no fence with ${target} tokens`);
  };

  it('returns no chunks for an empty body', async () => {
    expect(await chunkMarkdown('  \n\n ', 'T', md64)).toEqual([]);
  });

  it('rejects input containing the reserved sentinel characters', async () => {
    await expect(chunkMarkdown('a \uE000 b', 'T', md64)).rejects.toThrow(/U\+E000/);
  });

  it('keeps a fence just under the protection threshold whole', async () => {
    const fence = fenceWith(63);
    const chunks = await chunkMarkdown(`Intro text.\n\n${fence}\n\nOutro text.`, 'T', md64);
    expect(chunks.map((c) => c.content)).toContain(fence);
  });

  it('splits a fence at the threshold without leaving sentinel characters', async () => {
    const chunks = await chunkMarkdown(`Intro text.\n\n${fenceWith(64)}\n\nOutro text.`, 'T', md64);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.content).not.toMatch(/[\uE000\uE001]/);
      expect(c.tokenCount).toBeLessThanOrEqual(64);
    }
  });

  it('splits an oversized fence and stays within the chunk size', async () => {
    const code = Array.from({ length: 60 }, (_, i) => `const value${i} = compute(${i});`).join('\n');
    const chunks = await chunkMarkdown(`## Big\n\n\`\`\`ts\n${code}\n\`\`\``, 'T', md64);
    expect(chunks.length).toBeGreaterThan(5);
    for (const c of chunks) {
      expect(c.tokenCount).toBeLessThanOrEqual(64);
      expect(c.headingPath).toBe('T > Big');
    }
  });

  it('gives repeated text the heading path of its own position', async () => {
    const para = [
      'The first sentence of this paragraph is long enough to fill a good part of a small chunk on its own.',
      'The second sentence is repeated verbatim under both headings so that a naive search could match twice.',
      'The third sentence closes the paragraph and pushes the section well beyond the configured chunk size.',
      'The fourth sentence makes sure that the section really cannot fit into a single chunk of this size.',
    ].join('\n\n');
    const chunks = await chunkMarkdown(`## Alpha\n\n${para}\n\n## Beta\n\n${para}`, 'T', md64);
    const repeated = chunks.filter((c) => !c.content.startsWith('##'));
    expect(repeated.length).toBe(2);
    expect(repeated[0]?.content).toBe(repeated[1]?.content);
    expect(repeated.map((c) => c.headingPath)).toEqual(['T > Alpha', 'T > Beta']);
  });

  it('lets an H1 replace the title as root', async () => {
    const chunks = await chunkMarkdown('# Real Title\n\nText.', 'Fallback', md64);
    expect(chunks[0]?.headingPath).toBe('Real Title');
  });
});

describe('chunkDocument (M19b)', () => {
  const description = 'Widget is a fictional tool for synthetic tests.';
  const paragraphs = Array.from(
    { length: 12 },
    (_, i) => `Paragraph ${i} explains one more synthetic widget detail in plain words.`,
  );
  const doc = { title: 'Widget', body: `# Widget\n\n${paragraphs.join('\n\n')}`, description };
  const mdfm64 = parseChunkConfigKey('mdfm-64-8');

  it('opens chunk 0 with the description, and no other chunk', async () => {
    const chunks = await chunkDocument(doc, mdfm64);
    expect(chunks.length).toBeGreaterThan(2);
    expect(chunks[0]?.content.startsWith(`${description}\n\n# Widget`)).toBe(true);
    expect(chunks[0]?.headingPath).toBe('Widget');
    expect(chunks.slice(1).some((c) => c.content.includes(description))).toBe(false);
  });

  it('counts the description toward the chunk size', async () => {
    // Several sizes, so the description lands both in a chunk with room and in one it pushes text out of; every chunk
    // stays within the limit and no text is lost or duplicated (overlap 0).
    for (const size of [64, 96, 128]) {
      const config = parseChunkConfigKey(`mdfm-${size}-0`);
      const chunks = await chunkDocument(doc, config);
      for (const c of chunks) {
        expect(c.tokenCount).toBeLessThanOrEqual(size);
        expect(countTokens(c.content)).toBe(c.tokenCount);
      }
      const plain = await chunkDocument(doc, parseChunkConfigKey(`md-${size}-0`));
      expect(chunks.map((c) => c.content).join('\n\n')).toBe(
        `${description}\n\n${plain.map((c) => c.content).join('\n\n')}`,
      );
    }
  });

  it('equals the md chunks without a description, and ignores it for md keys', async () => {
    const plain = await chunkDocument(doc, md64);
    expect(await chunkDocument({ ...doc, description: undefined }, mdfm64)).toEqual(plain);
    expect(await chunkDocument({ ...doc, description: '' }, mdfm64)).toEqual(plain);
    expect(plain).toEqual(await chunkMarkdown(doc.body, doc.title, md64));
  });
});
