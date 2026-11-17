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

describe('expand (M16)', () => {
  it('reads result files without config.expand as off', async () => {
    const all = await loadResults(FIXTURES);
    expect(all.every((r) => r.data.config.expand === 'off')).toBe(true);
  });

  it('renders an expand matrix with off/on rows and labels a single config with expand=on', async () => {
    const all = await loadResults(FIXTURES);
    const base = all.find((r) => r.file.includes('baseline_retrieval') && !r.data.gitDirty);
    if (!base) {
      throw new Error('fixture missing');
    }
    const variant = (expand: 'off' | 'on', startedAt: string) => ({
      file: `${startedAt}_exp-expand_${expand}_retrieval.json`,
      data: {
        ...base.data,
        experimentId: 'exp-expand',
        startedAt,
        variedKey: 'expand' as const,
        config: { ...base.data.config, expand },
      },
    });
    const single = {
      file: 'z_single.json',
      data: {
        ...base.data,
        experimentId: 'single-expand',
        startedAt: '2026-02-01T10:00:02.000Z',
        config: { ...base.data.config, expand: 'on' as const },
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
    expect(block).toContain(`### exp-expand: expand\n\nFixed: ${chunkConfig}, ${embed}, k=${k}, ${search}.`);
    expect(block).toMatch(/\| expand \| Recall@k [^\n]*\n[^\n]*\n\| off \|[^\n]*\n\| on \|/);
    expect(block).toContain(`single-expand (${chunkConfig}, ${embed}, k=${k}, ${search}, expand=on)`);
  });
});

describe('rerank (M17)', () => {
  it('reads result files without config.rerank as off and labels a single config with rerank=on', async () => {
    const all = await loadResults(FIXTURES);
    expect(all.every((r) => r.data.config.rerank === 'off')).toBe(true);
    const base = all.find((r) => r.file.includes('baseline_retrieval') && !r.data.gitDirty);
    if (!base) {
      throw new Error('fixture missing');
    }
    const variant = (rerank: 'off' | 'on', startedAt: string) => ({
      file: `${startedAt}_exp-rerank_${rerank}_retrieval.json`,
      data: {
        ...base.data,
        experimentId: 'exp-rerank',
        startedAt,
        variedKey: 'rerank' as const,
        config: { ...base.data.config, rerank },
      },
    });
    const single = {
      file: 'z_single.json',
      data: {
        ...base.data,
        experimentId: 'single-rerank',
        startedAt: '2026-02-01T10:00:02.000Z',
        config: { ...base.data.config, rerank: 'on' as const },
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
    expect(block).toContain(`### exp-rerank: rerank\n\nFixed: ${chunkConfig}, ${embed}, k=${k}, ${search}.`);
    expect(block).toMatch(/\| rerank \| Recall@k [^\n]*\n[^\n]*\n\| off \|[^\n]*\n\| on \|/);
    expect(block).toContain(`single-rerank (${chunkConfig}, ${embed}, k=${k}, ${search}, rerank=on)`);
  });
});

describe('pipeline and deltas (M19)', () => {
  async function answersBase() {
    const all = await loadResults(FIXTURES);
    const base = all.find((r) => r.file.includes('baseline_answers') && !r.data.gitDirty);
    if (base?.data.evalType !== 'answers') {
      throw new Error('fixture missing');
    }
    return base.data;
  }

  it('reads result files without config.pipeline as baseline', async () => {
    const all = await loadResults(FIXTURES);
    expect(all.every((r) => r.data.config.pipeline === 'baseline')).toBe(true);
  });

  it('pairs a technique with its own off row, else with the newest same-config run, and renders the deltas', async () => {
    const base = await answersBase();
    const run = (
      experimentId: string,
      startedAt: string,
      over: Partial<typeof base.config>,
      variedKey: 'rerank' | 'pipeline' | null,
      metrics: { accuracy: number; p50: number },
    ) => ({
      file: `${startedAt}_${experimentId}_answers.json`,
      data: {
        ...base,
        experimentId,
        startedAt,
        variedKey,
        config: { ...base.config, ...over },
        aggregate: { all: { ...base.aggregate.all, accuracy: metrics.accuracy } },
        kpis: { ...base.kpis, latencyMs: { ...base.kpis.latencyMs, p50: metrics.p50 } },
      },
    });
    const block = renderBlock(
      selectResults([
        // The fixture baseline (2026-01-02, accuracy 4.25) has the same config but is older than exp-rerank off.
        ...(await loadResults(FIXTURES)),
        run('exp-rerank', '2026-02-01T10:00:00.000Z', { rerank: 'off' }, 'rerank', {
          accuracy: 4.5,
          p50: 2000,
        }),
        run('exp-rerank', '2026-02-01T10:00:01.000Z', { rerank: 'on' }, 'rerank', {
          accuracy: 4.6,
          p50: 9000,
        }),
        run('exp-agent', '2026-02-02T10:00:00.000Z', { pipeline: 'agent' }, 'pipeline', {
          accuracy: 4.25,
          p50: 7000,
        }),
      ]),
    );
    expect(block).toContain('### exp-agent: pipeline');
    expect(block).toMatch(/\| agent \| – \| – \| – \| – \| 4\.25 \|/);
    expect(block).toContain('| exp-agent: pipeline=agent | ');
    const deltas = block.slice(block.indexOf('### Phase 2 deltas'), block.indexOf('### KPIs'));
    expect(deltas).toMatchSnapshot();
    expect(deltas).toContain('| exp-rerank: rerank=on | exp-rerank: rerank=off | – | – | +0.10 |');
    expect(deltas).toContain('| exp-agent: pipeline=agent | exp-rerank: rerank=off | – | – | -0.25 |');
  });

  it('renders no delta table without a varied technique', async () => {
    expect(renderBlock(await selection())).not.toContain('Phase 2 deltas');
  });

  it('pairs an mdfm chunk config with the same md config from another experiment (M19b)', async () => {
    const all = await loadResults(FIXTURES);
    const retrieval = all.find((r) => r.file.includes('baseline_retrieval') && !r.data.gitDirty)?.data;
    const answers = await answersBase();
    if (retrieval?.evalType !== 'retrieval') {
      throw new Error('fixture missing');
    }
    const md = answers.config.chunkConfig;
    const mdfm = md.replace(/^md-/, 'mdfm-');
    const runs = [
      {
        id: 'exp-rerank',
        variedKey: 'rerank' as const,
        variant: 'off',
        at: '2026-02-01T10:00:0',
        recall: 0.7,
        acc: 4,
      },
      {
        id: 'exp-frontmatter',
        variedKey: 'chunkConfig' as const,
        variant: mdfm,
        at: '2026-02-03T10:00:0',
        recall: 0.75,
        acc: 4.2,
      },
    ].flatMap((r) => [
      {
        file: `${r.at}0Z_${r.id}_${r.variant}_retrieval.json`,
        data: {
          ...retrieval,
          experimentId: r.id,
          startedAt: `${r.at}0.000Z`,
          variedKey: r.variedKey,
          config: { ...retrieval.config, chunkConfig: r.variedKey === 'chunkConfig' ? mdfm : md },
          aggregate: { all: { ...retrieval.aggregate.all, recall: r.recall } },
        },
      },
      {
        file: `${r.at}1Z_${r.id}_${r.variant}_answers.json`,
        data: {
          ...answers,
          experimentId: r.id,
          startedAt: `${r.at}1.000Z`,
          variedKey: r.variedKey,
          config: { ...answers.config, chunkConfig: r.variedKey === 'chunkConfig' ? mdfm : md },
          aggregate: { all: { ...answers.aggregate.all, accuracy: r.acc } },
        },
      },
    ]);
    const block = renderBlock(selectResults([...all, ...runs]));
    expect(block).toContain('### exp-frontmatter: chunkConfig');
    const deltas = block.slice(block.indexOf('### Phase 2 deltas'), block.indexOf('### KPIs'));
    expect(deltas).toMatchSnapshot();
    expect(deltas).toContain(`| exp-frontmatter: chunkConfig=${mdfm} | exp-rerank: rerank=off | +0.05 |`);
    // md-only chunk sizes are no technique: exp-chunk-size (fixture) gets no delta row.
    expect(deltas).not.toContain('exp-chunk-size');
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
