import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  END,
  loadResults,
  pipelineKpis,
  renderBlock,
  replaceBlock,
  runReport,
  START,
  selectResults,
} from '../../src/cli/report';

const FIXTURES = join(import.meta.dirname, '../fixtures/results');

async function selection() {
  return selectResults(await loadResults(FIXTURES));
}

describe('selectResults', () => {
  it('keeps test-split runs only, latest per variant and eval type, and joins retrieval with answers', async () => {
    const sel = await selection();
    expect(sel.experiments.map((e) => e.id)).toEqual(['baseline', 'exp-chunk-size', 'exp-k']);
    const [baseline] = sel.experiments;
    expect(baseline?.variedKey).toBeNull();
    expect(baseline?.rows).toHaveLength(1);
    const row = baseline?.rows[0];
    // 2026-01-02 (recall 0.8, written without variedKey) beats 2026-01-01 (0.5); the dev run (0.1) is out.
    expect(row?.retrieval?.aggregate.all.recall).toBe(0.8);
    expect(row?.answers?.aggregate.all.accuracy).toBe(4.25);
  });

  it('ignores dirty, stale-base and other-dataset runs with a warning', async () => {
    const sel = await selection();
    expect(sel.warnings).toEqual([
      expect.stringMatching(/^2026-01-04T10-00-00Z_baseline_answers\.json: ignored, .*gitDirty/),
      expect.stringMatching(
        /^2025-12-01T10-00-00Z_exp-old_retrieval\.json: ignored, other dataset or corpus/,
      ),
      expect.stringMatching(
        /^2026-01-03T12-00-00Z_exp-chunk-size_md-256-32_retrieval\.json: ignored, fixed config/,
      ),
    ]);
    expect(sel.experiments.find((e) => e.id === 'exp-old')).toBeUndefined();
    const chunk = sel.experiments.find((e) => e.id === 'exp-chunk-size');
    expect(chunk?.rows.find((r) => r.variant === 'md-256-32')?.retrieval?.aggregate.all.recall).toBe(0.6);
  });

  it('orders matrix variants naturally', async () => {
    const sel = await selection();
    const variants = (id: string) => sel.experiments.find((e) => e.id === id)?.rows.map((r) => r.variant);
    expect(variants('exp-chunk-size')).toEqual(['md-256-32', 'md-512-64', 'md-1024-128']);
    expect(variants('exp-k')).toEqual(['3', '10']);
  });

  it('is independent of the listing order', async () => {
    const results = await loadResults(FIXTURES);
    expect(renderBlock(selectResults([...results].reverse()))).toBe(renderBlock(selectResults(results)));
  });

  it('reports nothing for dev-only results', async () => {
    const dev = (await loadResults(FIXTURES)).filter((r) => r.data.config.split === 'dev');
    const sel = selectResults(dev);
    expect(sel.experiments).toEqual([]);
    expect(renderBlock(sel)).toContain('No test-split results yet.');
  });
});

describe('pipelineKpis', () => {
  it('counts the pipeline without the judge, per question and call-weighted', async () => {
    const sel = await selection();
    const answers = sel.experiments[0]?.rows[0]?.answers;
    if (!answers) {
      throw new Error('fixture missing');
    }
    expect(pipelineKpis(answers)).toEqual({
      inputTokensPerQ: 2020,
      outputTokensPerQ: 50,
      costPer100: 0.2,
      estimatedTokenShare: 0.25,
      retryRate: 0.05,
      invalidOutputRate: 0.05,
    });
  });

  it('turns a missing token average into null instead of NaN', async () => {
    const sel = await selection();
    const answers = sel.experiments.find((e) => e.id === 'exp-k')?.rows[0]?.answers;
    expect(answers && pipelineKpis(answers).inputTokensPerQ).toBeNull();
  });
});

describe('renderBlock', () => {
  it('renders the main, matrix and KPI tables', async () => {
    expect(renderBlock(await selection())).toMatchSnapshot();
  });
});

describe('rewrite (M15)', () => {
  it('reads result files without config.rewrite as off', async () => {
    const all = await loadResults(FIXTURES);
    expect(all.length).toBeGreaterThan(0);
    expect(all.every((r) => r.data.config.rewrite === 'off')).toBe(true);
  });

  it('renders a rewrite matrix with off/on rows and labels a single config with rewrite=on', async () => {
    const all = await loadResults(FIXTURES);
    const base = all.find((r) => r.file.includes('baseline_retrieval') && !r.data.gitDirty);
    if (!base) {
      throw new Error('fixture missing');
    }
    const variant = (rewrite: 'off' | 'on', startedAt: string) => ({
      file: `${startedAt}_exp-rewrite_${rewrite}_retrieval.json`,
      data: {
        ...base.data,
        experimentId: 'exp-rewrite',
        startedAt,
        variedKey: 'rewrite' as const,
        config: { ...base.data.config, rewrite },
      },
    });
    const single = {
      file: 'z_single.json',
      data: {
        ...base.data,
        experimentId: 'single-rewrite',
        startedAt: '2026-02-01T10:00:02.000Z',
        config: { ...base.data.config, rewrite: 'on' as const },
      },
    };
    const block = renderBlock(
      selectResults([
        variant('on', '2026-02-01T10:00:01.000Z'),
        variant('off', '2026-02-01T10:00:00.000Z'),
        single,
      ]),
    );
    const { chunkConfig, embed, k, search } = base.data.config;
    expect(block).toContain(
      `### exp-rewrite: rewrite\n\nFixed: ${chunkConfig}, ${embed}, k=${k}, ${search}.`,
    );
    expect(block).toMatch(/\| rewrite \| Recall@k [^\n]*\n[^\n]*\n\| off \|[^\n]*\n\| on \|/);
    expect(block).toContain(`single-rewrite (${chunkConfig}, ${embed}, k=${k}, ${search}, rewrite=on)`);
  });
});

describe('replaceBlock', () => {
  it('replaces only the text between the markers', () => {
    const readme = `# Title\n\n${START}\nold\n${END}\n\nTail\n`;
    expect(replaceBlock(readme, '\nnew\n')).toBe(`# Title\n\n${START}\nnew\n${END}\n\nTail\n`);
  });

  it('refuses missing, duplicated or swapped markers', () => {
    expect(() => replaceBlock('# Title\n', '\n')).toThrow(/exactly one marker pair/);
    expect(() => replaceBlock(`${START}\n${END}\n${START}\n`, '\n')).toThrow(/exactly one marker pair/);
    expect(() => replaceBlock(`${END}\n${START}\n`, '\n')).toThrow(/exactly one marker pair/);
  });
});

describe('runReport', () => {
  let dir: string;
  let readmePath: string;
  const errors: string[] = [];

  beforeEach(async () => {
    dir = await mkdtemp(join(tmpdir(), 'report-'));
    readmePath = join(dir, 'README.md');
    errors.length = 0;
    vi.spyOn(console, 'error').mockImplementation((msg: string) => {
      errors.push(msg);
    });
    vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(async () => {
    vi.restoreAllMocks();
    await rm(dir, { recursive: true, force: true });
  });

  it('writes the block, then --check passes without writing', async () => {
    await writeFile(readmePath, `# Title\n\n${START}\n${END}\n`);
    expect(await runReport({ resultsDir: FIXTURES, readmePath, check: false })).toBe(0);
    const written = await readFile(readmePath, 'utf8');
    expect(written).toContain('| baseline (md-512-64, nomic-embed-text, k=5, vector) | 0.80 |');
    expect(await runReport({ resultsDir: FIXTURES, readmePath, check: true })).toBe(0);
    expect(await readFile(readmePath, 'utf8')).toBe(written);
  });

  it('--check fails on a hand-edited block and names the fix', async () => {
    await writeFile(readmePath, `# Title\n\n${START}\n${END}\n`);
    await runReport({ resultsDir: FIXTURES, readmePath, check: false });
    const edited = (await readFile(readmePath, 'utf8')).replace('| 0.80 |', '| 0.99 |');
    await writeFile(readmePath, edited);
    expect(await runReport({ resultsDir: FIXTURES, readmePath, check: true })).toBe(1);
    expect(errors.at(-1)).toMatch(/results block is out of date: run bun run report/);
    expect(await readFile(readmePath, 'utf8')).toBe(edited);
  });

  it('fails without markers and says which lines to add', async () => {
    await writeFile(readmePath, '# Title\n');
    expect(await runReport({ resultsDir: FIXTURES, readmePath, check: true })).toBe(1);
    expect(errors.at(-1)).toContain(`${START}\n${END}`);
  });

  it('fails on a file that is no result file', async () => {
    const results = join(dir, 'results');
    await mkdir(results);
    await writeFile(join(results, 'x.json'), '{"evalType":"retrieval"}');
    await writeFile(readmePath, `${START}\n${END}\n`);
    expect(await runReport({ resultsDir: results, readmePath, check: true })).toBe(1);
    expect(errors.at(-1)).toMatch(/x\.json is no result file/);
  });
});
