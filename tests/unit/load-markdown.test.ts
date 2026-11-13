import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { listMarkdownFiles, loadMarkdown, parseMarkdown } from '../../src/ingest/loadMarkdown';

const fixtures = join(import.meta.dirname, '../fixtures/docs');

describe('parseMarkdown', () => {
  it('takes the title from frontmatter, then the first H1, then the fallback', () => {
    expect(parseMarkdown('---\ntitle: "From FM"\n---\n# Heading\n', 'stem').title).toBe('From FM');
    expect(parseMarkdown("---\ntitle: 'Single'\n---\nText\n", 'stem').title).toBe('Single');
    expect(parseMarkdown('Intro\n\n# First H1\n\n## Sub\n', 'stem').title).toBe('First H1');
    expect(parseMarkdown('```sh\n# comment, not a title\n```\n', 'stem').title).toBe('stem');
  });

  it('removes frontmatter and normalizes CRLF', () => {
    const { body } = parseMarkdown(
      '---\r\ntitle: T\r\ndescription: D\r\n---\r\n\r\nLine one\r\nLine two\r\n',
      's',
    );
    expect(body).toBe('Line one\nLine two');
  });

  it('strips MDX imports, comments and component tags but keeps their children', () => {
    const raw = [
      'import { Note } from "/snippets/a.jsx";',
      '',
      '{/* hidden */}',
      '<Note>',
      '  Kept child text.',
      '</Note>',
      '<Accordion',
      '  title="Multi line"',
      '  open={true}',
      '>',
      'Accordion body.',
      '</Accordion>',
      '<Badge label="x" />',
      'Plain <b>html</b> stays.',
    ].join('\n');
    expect(parseMarkdown(raw, 's').body).toBe(
      ['Kept child text.', '', 'Accordion body.', '', 'Plain <b>html</b> stays.'].join('\n'),
    );
  });

  it('strips tags whose quoted attributes contain angle brackets or backticks', () => {
    const raw =
      '<Step title="1. <script> handling">\nStep body.\n</Step>\n\t<Accordion title="Why `a` and not `b`?">\nWhy.';
    expect(parseMarkdown(raw, 's').body).toBe('Step body.\n\nWhy.');
  });

  it('keeps runs of blank lines inside fences unchanged', () => {
    const fence = '```js\nimport a from "a";\n\n\n\nconst b = 1;\n```';
    expect(parseMarkdown(`Before.\n\n\n\n${fence}\n\n\n\nAfter.`, 's').body).toBe(
      `Before.\n\n${fence}\n\nAfter.`,
    );
  });

  it('does not let a stray backtick shield tags in later paragraphs', () => {
    const raw = 'A stray ` backtick.\n\n<Note>\nBody.\n</Note>\n\nAnother ` here.';
    expect(parseMarkdown(raw, 's').body).toBe('A stray ` backtick.\n\nBody.\n\nAnother ` here.');
  });

  it('leaves JSX inside fences and inline code untouched', () => {
    const raw = 'Use `<Widget />` here.\n\n```tsx\nimport { App } from "./app";\n<App />\n```\n';
    expect(parseMarkdown(raw, 's').body).toBe(raw.trim());
  });

  it('cleans the fixture document', async () => {
    const doc = await loadMarkdown(fixtures, 'widget-guide.mdx');
    expect(doc.sourcePath).toBe('widget-guide.mdx');
    expect(doc.title).toBe('Widget Guide');
    for (const gone of ['import {', '{/*', '<Note>', '</Note>', '<Accordion', '<Badge', 'description:']) {
      expect(doc.body).not.toContain(gone);
    }
    expect(doc.body).toContain('Configuration lives in a plain text file');
    expect(doc.body).toContain('`<Widget />`');
    const raw = await readFile(join(fixtures, 'widget-guide.mdx'), 'utf8');
    const fence = raw.slice(raw.indexOf('```ts'), raw.indexOf('};\n```') + 5);
    expect(doc.body).toContain(fence);
  });
});

describe('listMarkdownFiles', () => {
  let root: string;

  beforeAll(async () => {
    root = await mkdtemp(join(tmpdir(), 'load-md-'));
    await mkdir(join(root, 'b', 'nested'), { recursive: true });
    await mkdir(join(root, 'a.md'));
    for (const f of ['z.mdx', 'b/nested/y.md', 'b/x.MDX', 'b/image.png', 'docs.json']) {
      await writeFile(join(root, f), 'x');
    }
  });

  afterAll(async () => {
    await rm(root, { recursive: true, force: true });
  });

  it('returns sorted relative .md/.mdx files only', async () => {
    expect(await listMarkdownFiles(root)).toEqual(['b/nested/y.md', 'b/x.MDX', 'z.mdx']);
    expect(await listMarkdownFiles(`${root}/`)).toEqual(['b/nested/y.md', 'b/x.MDX', 'z.mdx']);
  });

  it('returns correct paths for a relative root with ./ and ..', async () => {
    const rel = relative(process.cwd(), root);
    expect(await listMarkdownFiles(`./${rel}`)).toEqual(['b/nested/y.md', 'b/x.MDX', 'z.mdx']);
    expect(await listMarkdownFiles(join(rel, 'b', '..'))).toEqual(['b/nested/y.md', 'b/x.MDX', 'z.mdx']);
  });
});
