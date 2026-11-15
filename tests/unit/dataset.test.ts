import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

import { main } from '../../src/cli/evalValidate';
import {
  CATEGORIES,
  CATEGORY_TARGETS,
  checkSpans,
  corpusReader,
  countWarnings,
  devTarget,
  type GoldenEntry,
  type Issue,
  loadDataset,
  parseDataset,
} from '../../src/eval/dataset';
import { normalize } from '../../src/eval/normalize';
import { listCorpus } from '../../src/ingest/loadMarkdown';

const fixtures = join(import.meta.dirname, '../fixtures/eval');
const root = join(fixtures, 'docs');
const validFile = join(fixtures, 'valid.jsonl');
const brokenFile = join(fixtures, 'broken-span.jsonl');

const readBody = corpusReader(root, await listCorpus(root));

/** A valid answerable entry; overrides replace whole fields. */
function entry(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: 'q001',
    split: 'dev',
    category: 'direct_fact',
    question: 'Synthetic question?',
    answerable: true,
    evidence: [{ source: 'widget.mdx', span: 'The widget server listens on port 8080' }],
    keywords: ['8080'],
    reference_answer: 'Synthetic answer.',
    ...overrides,
  };
}

const jsonl = (...entries: Record<string, unknown>[]) => entries.map((e) => JSON.stringify(e)).join('\n');
const kinds = (issues: Issue[]) => issues.map((i) => i.kind);
const details = (issues: Issue[]) => issues.map((i) => i.detail).join('\n');

describe('normalize', () => {
  it('lowercases and collapses whitespace', () => {
    expect(normalize('  Foo\n\tBAR   baz  ')).toBe('foo bar baz');
  });

  it('removes inline markup and keeps link and image text', () => {
    expect(
      normalize('Set `WIDGET_HOME` to **bold** and *em* ~~old~~ [the docs](https://example.com/a)'),
    ).toBe('set widget_home to bold and em old the docs');
    expect(normalize('![Logo](https://example.com/logo.png) <https://example.com>')).toBe(
      'logo https://example.com',
    );
  });

  it('removes underscore emphasis but keeps underscores inside words', () => {
    expect(normalize('_note_ NODE_ENV snake_case __strong__')).toBe('note node_env snake_case strong');
  });

  it('removes block markers at line starts, fence lines and escapes', () => {
    expect(normalize('# Title\n- item one\n> quote\n> - nested\n2. step\n| a | b |')).toBe(
      'title item one quote nested step a b',
    );
    expect(normalize('```sh title="x"\nwidget start\n```')).toBe('widget start');
    expect(normalize('a \\* b \\[x\\]')).toBe('a b [x]');
  });

  it('is stable on its own output', () => {
    const text = '## Setup\n\n- Run `widget start` to launch the **server**.\n> Logs: _the logs directory_.';
    expect(normalize(normalize(text))).toBe(normalize(text));
  });

  it('does not match a span copied across a list-item break', () => {
    const body = normalize('- Run one\n- Run two');
    expect(body.includes(normalize('Run one Run two'))).toBe(true);
    expect(body.includes(normalize('Run one - Run two'))).toBe(false);
  });

  it('with fromLineStart false keeps markers on the first line only, as mid-line text', () => {
    expect(normalize('2. step\n- item\n# Title', { fromLineStart: false })).toBe('2. step item title');
    expect(normalize('> - nested', { fromLineStart: false })).toBe('> - nested');
    // A fence-like run on the first line is not a fence line; its backticks go as inline markup.
    expect(normalize('```sh\nwidget start\n```', { fromLineStart: false })).toBe('sh widget start');
  });
});

describe('parseDataset', () => {
  it('accepts the valid fixture', async () => {
    const parsed = parseDataset(await readFile(validFile, 'utf8'));
    expect(parsed.issues).toEqual([]);
    expect(parsed.entries.map((e) => e.id)).toEqual(['q001', 'q002', 'q003']);
    expect(parsed.lines).toBe(3);
  });

  it('skips blank lines and reports invalid JSON with its line number', () => {
    const parsed = parseDataset(`${jsonl(entry())}\n\n{not json\n`);
    expect(parsed.lines).toBe(2);
    expect(parsed.issues).toMatchObject([{ level: 'error', ref: 'line 3', kind: 'invalid_json' }]);
  });

  it('rejects unknown keys, a bad id and a non-llm origin', () => {
    const typo = parseDataset(jsonl(entry({ refrence_answer: 'x' })));
    expect(kinds(typo.issues)).toEqual(['schema']);
    expect(parseDataset(jsonl(entry({ id: 'x1' }))).issues[0]).toMatchObject({ ref: 'x1', kind: 'schema' });
    expect(parseDataset(jsonl(entry({ origin: 'human' }))).issues).toHaveLength(1);
    expect(parseDataset(jsonl(entry({ origin: 'llm' }))).issues).toEqual([]);
  });

  it('ties negative, answerable=false and empty evidence together', () => {
    const negAnswerable = entry({ category: 'negative', evidence: [] });
    expect(details(parseDataset(jsonl(negAnswerable)).issues)).toMatch(/answerable: must be false/);
    const factUnanswerable = entry({ answerable: false, evidence: [] });
    expect(details(parseDataset(jsonl(factUnanswerable)).issues)).toMatch(/answerable: must be false/);
    const negWithEvidence = entry({ category: 'negative', answerable: false, keywords: [] });
    expect(details(parseDataset(jsonl(negWithEvidence)).issues)).toMatch(/evidence: must be empty/);
    const negOk = entry({ category: 'negative', answerable: false, evidence: [], keywords: [] });
    expect(parseDataset(jsonl(negOk)).issues).toEqual([]);
  });

  it('requires spans and keywords for answerable entries', () => {
    expect(details(parseDataset(jsonl(entry({ evidence: [] }))).issues)).toMatch(/at least one span/);
    expect(details(parseDataset(jsonl(entry({ keywords: [] }))).issues)).toMatch(/at least one keyword/);
  });

  it('requires two distinct sources for multi_document', () => {
    const sameSource = entry({
      category: 'multi_document',
      evidence: [
        { source: 'widget.mdx', span: 'one span' },
        { source: 'widget.mdx', span: 'other span' },
      ],
    });
    expect(details(parseDataset(jsonl(sameSource)).issues)).toMatch(/2 distinct sources/);
  });

  it('limits spans to 15 words and rejects duplicate or markup-only spans', () => {
    const words = (n: number) => Array.from({ length: n }, (_, i) => `w${i}`).join(' ');
    const span = (s: string) => entry({ evidence: [{ source: 'widget.mdx', span: s }] });
    expect(parseDataset(jsonl(span(words(15)))).issues).toEqual([]);
    expect(parseDataset(jsonl(span(words(16)))).issues[0]).toMatchObject({
      source: 'widget.mdx',
      detail: 'evidence.0.span: 16 words, at most 15',
    });
    expect(details(parseDataset(jsonl(span('**`**'))).issues)).toMatch(/empty after normalization/);
    const dup = entry({
      evidence: [
        { source: 'widget.mdx', span: 'Port 8080' },
        { source: 'widget.mdx', span: 'port  **8080**' },
      ],
    });
    expect(details(parseDataset(jsonl(dup)).issues)).toMatch(/same source and span/);
  });

  it('requires sources relative to the docs root', () => {
    const src = (source: string) => parseDataset(jsonl(entry({ evidence: [{ source, span: 'x' }] }))).issues;
    expect(details(src('docs/widget.mdx'))).toMatch(/drop the "docs\/" prefix/);
    expect(details(src('/abs/widget.mdx'))).toMatch(/relative to the docs root/);
    expect(details(src('../widget.mdx'))).toMatch(/plain relative path/);
    expect(details(src('widget.txt'))).toMatch(/\.md or \.mdx/);
  });

  it('reports a repeated id even when the first line is invalid', () => {
    const parsed = parseDataset(jsonl(entry({ split: 'nope' }), entry()));
    expect(kinds(parsed.issues)).toEqual(['schema', 'duplicate_id']);
  });

  it('does not count table pipes as span words', () => {
    const cells = Array.from({ length: 8 }, (_, i) => `c${i}`).join(' | ');
    expect(parseDataset(jsonl(entry({ evidence: [{ source: 'widget.mdx', span: cells }] }))).issues).toEqual(
      [],
    );
  });

  it('reports a repeated id and keeps the first entry', () => {
    const parsed = parseDataset(jsonl(entry(), entry({ question: 'Another?' })));
    expect(parsed.entries).toHaveLength(1);
    expect(parsed.issues).toMatchObject([
      { ref: 'q001', kind: 'duplicate_id', detail: 'line 2 repeats the id of line 1' },
    ]);
  });
});

describe('checkSpans', () => {
  const entriesOf = async (file: string) => parseDataset(await readFile(file, 'utf8')).entries;

  it('finds every span of the valid fixture', async () => {
    expect(await checkSpans(await entriesOf(validFile), readBody)).toEqual([]);
  });

  it('reports the broken span fixture', async () => {
    const issues = await checkSpans(await entriesOf(brokenFile), readBody);
    expect(issues).toMatchObject([{ ref: 'q001', source: 'widget.mdx', kind: 'span_not_found' }]);
    expect(issues[0]?.detail).toContain('port 9090');
  });

  it('checks against the loaded body, not the raw file', async () => {
    const parsed = parseDataset(
      jsonl(
        entry({
          id: 'q001',
          evidence: [{ source: 'widget.mdx', span: 'Frontmatter sentence that is never chunked' }],
        }),
        entry({
          id: 'q002',
          evidence: [{ source: 'widget.mdx', span: 'Attribute text that is never chunked' }],
        }),
        entry({
          id: 'q003',
          evidence: [{ source: 'widget.mdx', span: 'Logs are written to the logs directory.' }],
        }),
      ),
    );
    expect(kinds(await checkSpans(parsed.entries, readBody))).toEqual(['span_not_found', 'span_not_found']);
  });

  it('reports a missing source once per entry', async () => {
    const parsed = parseDataset(
      jsonl(
        entry({
          evidence: [
            { source: 'nope.md', span: 'a' },
            { source: 'nope.md', span: 'b' },
          ],
        }),
      ),
    );
    expect(await checkSpans(parsed.entries, readBody)).toMatchObject([
      { ref: 'q001', source: 'nope.md', kind: 'source_not_found' },
    ]);
  });

  it('requires the exact case of the source path', async () => {
    const parsed = parseDataset(
      jsonl(entry({ evidence: [{ source: 'Widget.mdx', span: 'The widget server listens on port 8080' }] })),
    );
    expect(await checkSpans(parsed.entries, readBody)).toMatchObject([
      { source: 'Widget.mdx', kind: 'source_not_found' },
    ]);
  });
});

describe('countWarnings', () => {
  it('splits the §6 targets 5/3/3/2/2 into dev', () => {
    expect(CATEGORIES.map((c) => devTarget(CATEGORY_TARGETS[c]))).toEqual([5, 3, 3, 2, 2]);
  });

  it('is silent for a complete, stratified dataset', () => {
    let n = 0;
    const entries = CATEGORIES.flatMap((category) =>
      Array.from({ length: CATEGORY_TARGETS[category] }, (_, i) => ({
        id: `q${String(++n).padStart(3, '0')}`,
        category,
        split: i < devTarget(CATEGORY_TARGETS[category]) ? 'dev' : 'test',
      })),
    ) as GoldenEntry[];
    expect(entries).toHaveLength(45);
    expect(countWarnings(entries)).toEqual([]);
  });

  it('warns about an incomplete dataset and LLM-written entries', () => {
    const entries = [
      { id: 'q001', category: 'direct_fact', split: 'test', origin: 'llm' },
      { id: 'q002', category: 'direct_fact', split: 'test' },
    ] as GoldenEntry[];
    const warnings = countWarnings(entries);
    expect(warnings.every((w) => w.level === 'warning')).toBe(true);
    expect(kinds(warnings)).toEqual([
      'count_total',
      'count_split',
      'count_category',
      'split_category',
      'count_category',
      'count_category',
      'count_category',
      'count_category',
      'origin_llm',
    ]);
    expect(details(warnings)).toContain('direct_fact: dev 0 of 2, expected 1');
  });
});

describe('loadDataset', () => {
  it('hashes the raw file bytes and returns undefined for a missing file', async () => {
    const loaded = await loadDataset(validFile);
    expect(loaded?.hash).toBe(
      createHash('sha256')
        .update(await readFile(validFile))
        .digest('hex'),
    );
    expect(await loadDataset(join(fixtures, 'absent.jsonl'))).toBeUndefined();
  });
});

describe('eval:validate CLI', () => {
  let tmp: string;
  let out: string[];
  let err: string[];

  beforeAll(async () => {
    tmp = await mkdtemp(join(tmpdir(), 'eval-validate-'));
  });
  afterAll(async () => {
    await rm(tmp, { recursive: true, force: true });
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  async function run(...argv: string[]): Promise<number> {
    out = [];
    err = [];
    vi.spyOn(console, 'log').mockImplementation((m: string) => void out.push(m));
    vi.spyOn(console, 'error').mockImplementation((m: string) => void err.push(m));
    return main(argv);
  }

  it('reports a missing or empty dataset', async () => {
    expect(await run('--file', join(tmp, 'absent.jsonl'), '--root', root)).toBe(1);
    expect(err.join('\n')).toMatch(/absent\.jsonl missing or empty/);
    const empty = join(tmp, 'empty.jsonl');
    await writeFile(empty, '\n');
    expect(await run('--file', empty, '--root', root)).toBe(1);
    expect(err.join('\n')).toMatch(/missing or empty/);
  });

  it('passes the valid fixture with count warnings, and fails it under --strict', async () => {
    expect(await run('--file', validFile, '--root', root)).toBe(0);
    expect(out.at(-1)).toMatch(
      /^3 entries, 3 valid \(1 dev \/ 2 test\): 0 errors, \d+ warnings; sha256 [0-9a-f]{64}$/,
    );
    expect(out.some((l) => l.startsWith('WARN   -  -  count_split  dev 1/15, test 2/30'))).toBe(true);
    expect(await run('--file', validFile, '--root', root, '--strict')).toBe(1);
    expect(out.some((l) => l.startsWith('ERROR  -  -  count_total'))).toBe(true);
  });

  it('prints one line per broken span', async () => {
    expect(await run('--file', brokenFile, '--root', root)).toBe(1);
    expect(out[0]).toBe(
      'ERROR  q001  widget.mdx  span_not_found  "The widget server listens on port 9090 by default."',
    );
  });

  it('says how to fetch a missing corpus', async () => {
    expect(await run('--file', validFile, '--root', join(tmp, 'no-corpus'))).toBe(1);
    expect(err.join('\n')).toMatch(
      /Corpus directory .*no-corpus not found; run \.\/scripts\/fetch-corpus\.sh \(spans not checked\)/,
    );
  });

  it('reports an empty corpus instead of a missing source per entry', async () => {
    const empty = join(tmp, 'empty-corpus');
    await mkdir(empty);
    expect(await run('--file', validFile, '--root', empty)).toBe(1);
    expect(out.some((l) => l.includes('source_not_found'))).toBe(false);
    expect(err.join('\n')).toMatch(
      /No \.md\/\.mdx files under .*empty-corpus; run \.\/scripts\/fetch-corpus\.sh/,
    );
  });

  it('exits 2 on a usage error', async () => {
    expect(await run('--nope')).toBe(2);
    expect(await run('extra')).toBe(2);
  });
});
