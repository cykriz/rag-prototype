// bun run report [--check]
// Renders the README results tables (docs/PLAN.md §11 item 8, §8 README KPIs) from results/*.json between
// <!-- results:start --> and <!-- results:end -->. --check writes nothing and exits 1 when README.md differs from
// what results/ generates. Only test-split runs from a clean checkout count (§6); per experiment, variant and eval
// type the latest run wins. File names are never parsed: everything comes from the JSON. Exit 0 when written or up
// to date, 1 on a difference, missing markers or an unreadable result file, 2 on a usage error.
import { existsSync } from 'node:fs';
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { z } from 'zod';
import { RunConfig, VARY_KEYS, type VaryKey } from '../eval/experiment';

export const START = '<!-- results:start -->';
export const END = '<!-- results:end -->';
const PLACEHOLDER = '–';
const USAGE = 'Usage: bun run report [--check]\n  --check  compare README.md with results/ without writing';

const Num = z.number().nullable();
const Header = {
  experimentId: z.string(),
  startedAt: z.string(),
  gitDirty: z.boolean(),
  datasetSha256: z.string(),
  corpusSha: z.string(),
  // The written config is exactly RunConfig (resultHeader); strict, so a new key cannot be dropped silently.
  config: RunConfig,
  // Added in M13a; earlier files are single-config runs.
  variedKey: z.enum(VARY_KEYS).nullable().default(null),
};
const RetrievalFile = z.object({
  ...Header,
  evalType: z.literal('retrieval'),
  aggregate: z.object({ all: z.object({ recall: Num, mrr: Num, ndcg: Num, keywordCoverage: Num }) }),
});
const AnswersFile = z.object({
  ...Header,
  evalType: z.literal('answers'),
  counts: z.object({ questions: z.number() }),
  aggregate: z.object({
    all: z.object({
      accuracy: Num,
      completeness: Num,
      relevance: Num,
      abstentionAccuracy: Num,
      falseAbstentionRate: Num,
    }),
  }),
  kpis: z.object({
    latencyMs: z.object({ p50: Num, p95: Num }),
    calls: z.array(
      z.object({
        operation: z.string(),
        calls: z.number(),
        avgInputTokens: Num,
        avgOutputTokens: Num,
        estimatedTokenShare: z.number(),
        retryRate: z.number(),
        invalidOutputRate: z.number(),
        costUsd: z.number(),
      }),
    ),
  }),
});
const ResultFile = z.discriminatedUnion('evalType', [RetrievalFile, AnswersFile]);
export type RetrievalFile = z.infer<typeof RetrievalFile>;
export type AnswersFile = z.infer<typeof AnswersFile>;
export type ResultFile = z.infer<typeof ResultFile>;

export interface LoadedResult {
  file: string;
  data: ResultFile;
}

/** Every `*.json` in `dir`, sorted by name; a missing dir is empty. Throws naming the file that fails the schema. */
export async function loadResults(dir: string): Promise<LoadedResult[]> {
  if (!existsSync(dir)) {
    return [];
  }
  const names = (await readdir(dir)).filter((n) => n.endsWith('.json')).sort();
  const out: LoadedResult[] = [];
  for (const file of names) {
    let raw: unknown;
    try {
      raw = JSON.parse(await readFile(join(dir, file), 'utf8'));
    } catch (err) {
      throw new Error(`${join(dir, file)}: ${err instanceof Error ? err.message : String(err)}`);
    }
    const parsed = ResultFile.safeParse(raw);
    if (!parsed.success) {
      const issues = parsed.error.issues
        .map((i) => `${i.path.join('.') || '(root)'}: ${i.message}`)
        .join('; ');
      throw new Error(`${join(dir, file)} is no result file: ${issues}`);
    }
    out.push({ file, data: parsed.data });
  }
  return out;
}

type Config = ResultFile['config'];

export interface Row {
  /** The varied value; null in a single-config experiment. */
  variant: string | null;
  config: Config;
  retrieval: RetrievalFile | null;
  answers: AnswersFile | null;
}

export interface Experiment {
  id: string;
  variedKey: VaryKey | null;
  rows: Row[];
}

export interface Selection {
  /** In greedy order: by the earliest startedAt of their selected files, then id. */
  experiments: Experiment[];
  datasetSha256: string | null;
  corpusSha: string | null;
  /** Ignored files with the reason, for stderr; never part of the README block. */
  warnings: string[];
}

/** Later startedAt wins, the file name breaks ties, so the choice is independent of the listing order. */
const newer = (a: LoadedResult, b: LoadedResult) =>
  a.data.startedAt === b.data.startedAt ? a.file > b.file : a.data.startedAt > b.data.startedAt;
const latest = (rs: readonly LoadedResult[]) =>
  rs.reduce<LoadedResult | undefined>((m, r) => (!m || newer(r, m) ? r : m), undefined);

const variantOf = (d: ResultFile) => (d.variedKey === null ? null : String(d.config[d.variedKey]));

/** The config an experiment holds fixed: every key except the varied one and the split, plus the varied key. */
function baseKey(d: ResultFile): string {
  const { split: _, ...fixed } = d.config;
  if (d.variedKey !== null && d.variedKey !== 'split') {
    delete fixed[d.variedKey];
  }
  return JSON.stringify({ variedKey: d.variedKey, ...fixed });
}

const collator = new Intl.Collator('en', { numeric: true });

/**
 * Test split only (§6); dirty runs are ignored, as are runs on another dataset or corpus than the newest clean one
 * and runs whose fixed config differs from the experiment's latest run (stale after a greedy step). Per experiment,
 * variant and eval type the latest run wins; retrieval and answers of one variant form one row.
 */
export function selectResults(results: readonly LoadedResult[]): Selection {
  const warnings: string[] = [];
  const test = results.filter((r) => r.data.config.split === 'test');
  const clean: LoadedResult[] = [];
  for (const r of test) {
    if (r.data.gitDirty) {
      warnings.push(`${r.file}: ignored, run from a checkout with uncommitted changes (gitDirty)`);
    } else {
      clean.push(r);
    }
  }
  const newest = latest(clean);
  if (!newest) {
    return { experiments: [], datasetSha256: null, corpusSha: null, warnings };
  }
  const { datasetSha256, corpusSha } = newest.data;
  const current = clean.filter((r) => {
    const same = r.data.datasetSha256 === datasetSha256 && r.data.corpusSha === corpusSha;
    if (!same) {
      warnings.push(`${r.file}: ignored, other dataset or corpus than the newest run (${newest.file})`);
    }
    return same;
  });

  const byExperiment = Map.groupBy(current, (r) => r.data.experimentId);
  const experiments: (Experiment & { order: string })[] = [];
  for (const [id, runs] of byExperiment) {
    const head = latest(runs);
    if (!head) {
      continue;
    }
    const base = baseKey(head.data);
    const kept = runs.filter((r) => {
      const same = baseKey(r.data) === base;
      if (!same) {
        warnings.push(`${r.file}: ignored, fixed config differs from the latest ${id} run (${head.file})`);
      }
      return same;
    });
    const rows: Row[] = [];
    const selected: LoadedResult[] = [];
    for (const [variant, group] of Map.groupBy(kept, (r) => variantOf(r.data))) {
      const pick = (type: ResultFile['evalType']) => latest(group.filter((r) => r.data.evalType === type));
      const retrieval = pick('retrieval');
      const answers = pick('answers');
      const config = (retrieval ?? answers ?? head).data.config;
      rows.push({
        variant,
        config,
        retrieval: retrieval?.data.evalType === 'retrieval' ? retrieval.data : null,
        answers: answers?.data.evalType === 'answers' ? answers.data : null,
      });
      selected.push(...[retrieval, answers].filter((r) => r !== undefined));
    }
    rows.sort((a, b) => collator.compare(a.variant ?? '', b.variant ?? ''));
    const order = selected.map((r) => r.data.startedAt).sort()[0] ?? '';
    experiments.push({ id, variedKey: head.data.variedKey, rows, order });
  }
  experiments.sort((a, b) => (a.order === b.order ? a.id.localeCompare(b.id) : a.order < b.order ? -1 : 1));
  return { experiments: experiments.map(({ order: _, ...e }) => e), datasetSha256, corpusSha, warnings };
}

const fixed2 = (x: number | null | undefined) => (x === null || x === undefined ? PLACEHOLDER : x.toFixed(2));
const integer = (x: number | null | undefined) =>
  x === null || x === undefined ? PLACEHOLDER : String(Math.round(x));

export interface PipelineKpis {
  inputTokensPerQ: number | null;
  outputTokensPerQ: number | null;
  costPer100: number | null;
  estimatedTokenShare: number | null;
  retryRate: number | null;
  invalidOutputRate: number | null;
}

/**
 * KPIs of the answer pipeline: every llm_calls group except the judge, which is eval overhead. Token and cost sums
 * are per question; shares are weighted by calls. A null average token count makes the token figures null.
 */
export function pipelineKpis(a: AnswersFile): PipelineKpis {
  const calls = a.kpis.calls.filter((c) => c.operation !== 'judge');
  const q = a.counts.questions;
  const n = calls.reduce((s, c) => s + c.calls, 0);
  const perQ = (field: 'avgInputTokens' | 'avgOutputTokens') =>
    q === 0 || calls.some((c) => c[field] === null)
      ? null
      : calls.reduce((s, c) => s + c.calls * (c[field] ?? 0), 0) / q;
  const share = (field: 'estimatedTokenShare' | 'retryRate' | 'invalidOutputRate') =>
    n === 0 ? null : calls.reduce((s, c) => s + c.calls * c[field], 0) / n;
  return {
    inputTokensPerQ: perQ('avgInputTokens'),
    outputTokensPerQ: perQ('avgOutputTokens'),
    costPer100: q === 0 ? null : (calls.reduce((s, c) => s + c.costUsd, 0) * 100) / q,
    estimatedTokenShare: share('estimatedTokenShare'),
    retryRate: share('retryRate'),
    invalidOutputRate: share('invalidOutputRate'),
  };
}

const METRIC_HEADERS = [
  'Recall@k',
  'MRR@k',
  'nDCG@k',
  'KW cov.',
  'Accuracy',
  'Completeness',
  'Relevance',
  'Abstention acc.',
  'p50 ms',
  'p95 ms',
  'Tokens/q',
  'Cost/100 q',
];

function metricCells(row: Row): string[] {
  const r = row.retrieval?.aggregate.all;
  const a = row.answers?.aggregate.all;
  const k = row.answers ? pipelineKpis(row.answers) : null;
  const tokens =
    k?.inputTokensPerQ == null || k.outputTokensPerQ == null ? null : k.inputTokensPerQ + k.outputTokensPerQ;
  return [
    fixed2(r?.recall),
    fixed2(r?.mrr),
    fixed2(r?.ndcg),
    fixed2(r?.keywordCoverage),
    fixed2(a?.accuracy),
    fixed2(a?.completeness),
    fixed2(a?.relevance),
    fixed2(a?.abstentionAccuracy),
    integer(row.answers?.kpis.latencyMs.p50),
    integer(row.answers?.kpis.latencyMs.p95),
    integer(tokens),
    fixed2(k?.costPer100),
  ];
}

const table = (headers: readonly string[], rows: readonly string[][]) =>
  [headers, headers.map(() => '---'), ...rows].map((cells) => `| ${cells.join(' | ')} |`).join('\n');

const describe = (c: Config, skip: VaryKey | null = null) =>
  [
    skip !== 'chunkConfig' && c.chunkConfig,
    skip !== 'embed' && c.embed,
    skip !== 'k' && `k=${c.k}`,
    skip !== 'search' && c.search,
    // Only when on, so rows of runs before M15 (read as off) render as before.
    skip !== 'rewrite' && c.rewrite === 'on' && 'rewrite=on',
  ]
    .filter(Boolean)
    .join(', ');

const rowLabel = (e: Experiment, row: Row) =>
  e.variedKey === null ? `${e.id} (${describe(row.config)})` : `${e.id}: ${e.variedKey}=${row.variant}`;

/** The text between the markers, starting and ending with a newline; pure, so --check compares exact strings. */
export function renderBlock(sel: Selection): string {
  const parts = ['<!-- Generated by `bun run report` from results/; do not edit by hand. -->'];
  if (sel.experiments.length === 0) {
    parts.push('No test-split results yet.');
    return `\n${parts.join('\n\n')}\n`;
  }
  const main = sel.experiments.filter((e) => e.variedKey === null);
  const matrices = sel.experiments.filter((e) => e.variedKey !== null);
  if (main.length > 0) {
    parts.push(
      table(
        ['Config', ...METRIC_HEADERS],
        main.flatMap((e) => e.rows.map((row) => [rowLabel(e, row), ...metricCells(row)])),
      ),
    );
  }
  for (const e of matrices) {
    const key = e.variedKey as VaryKey;
    const first = e.rows[0];
    parts.push(
      `### ${e.id}: ${key}`,
      ...(first ? [`Fixed: ${describe(first.config, key)}.`] : []),
      table(
        [key, ...METRIC_HEADERS],
        e.rows.map((row) => [row.variant ?? PLACEHOLDER, ...metricCells(row)]),
      ),
    );
  }
  parts.push('### KPIs');
  const kpiRows = sel.experiments.flatMap((e) =>
    e.rows.flatMap((row) => {
      if (!row.answers) {
        return [];
      }
      const k = pipelineKpis(row.answers);
      return [
        [
          rowLabel(e, row),
          integer(k.inputTokensPerQ),
          integer(k.outputTokensPerQ),
          fixed2(k.estimatedTokenShare),
          fixed2(k.retryRate),
          fixed2(k.invalidOutputRate),
          fixed2(row.answers.aggregate.all.falseAbstentionRate),
        ],
      ];
    }),
  );
  parts.push(
    kpiRows.length === 0
      ? 'No answers run on the test split yet.'
      : table(
          [
            'Config',
            'In tok/q',
            'Out tok/q',
            'Est. token share',
            'Retry rate',
            'Invalid output rate',
            'False abst.',
          ],
          kpiRows,
        ),
  );
  parts.push(
    'Test split only, latest run per config from a clean checkout. Metrics are the `all` group of each result ' +
      'file (human and llm reference answers are reported separately there); `–` = not run or not recorded. Tokens/q, Cost/100 q ' +
      `and the KPI rates count the answer pipeline without the judge. Dataset \`${sel.datasetSha256?.slice(0, 8)}\`, ` +
      `corpus \`${sel.corpusSha?.slice(0, 8)}\`.`,
  );
  return `\n${parts.join('\n\n')}\n`;
}

const count = (text: string, marker: string) => text.split(marker).length - 1;

/** `readme` with the text between the markers replaced; throws unless there is exactly one marker pair in order. */
export function replaceBlock(readme: string, block: string): string {
  const start = readme.indexOf(START);
  const end = readme.indexOf(END);
  if (count(readme, START) !== 1 || count(readme, END) !== 1 || end < start) {
    throw new Error(
      `README needs exactly one marker pair, in this order, on lines of their own:\n${START}\n${END}`,
    );
  }
  return readme.slice(0, start + START.length) + block + readme.slice(end);
}

/** Renders and writes README.md, or with `check` only compares; returns the exit code. */
export async function runReport(opts: {
  resultsDir: string;
  readmePath: string;
  check: boolean;
}): Promise<number> {
  let next: string;
  let readme: string;
  try {
    const selection = selectResults(await loadResults(opts.resultsDir));
    for (const w of selection.warnings) {
      console.error(`warning: ${w}`);
    }
    readme = await readFile(opts.readmePath, 'utf8');
    next = replaceBlock(readme, renderBlock(selection));
  } catch (err) {
    console.error(`${opts.readmePath}: ${err instanceof Error ? err.message : String(err)}`);
    return 1;
  }
  if (opts.check) {
    if (next !== readme) {
      console.error(`${opts.readmePath} results block is out of date: run bun run report`);
      return 1;
    }
    console.log(`${opts.readmePath} results block is up to date`);
    return 0;
  }
  if (next !== readme) {
    await writeFile(opts.readmePath, next);
  }
  console.log(`${opts.readmePath} results block ${next === readme ? 'unchanged' : 'updated'}`);
  return 0;
}

export async function main(argv: string[]): Promise<number> {
  let check: boolean;
  try {
    const { values } = parseArgs({
      args: argv,
      strict: true,
      options: {
        check: { type: 'boolean', default: false },
        help: { type: 'boolean', short: 'h', default: false },
      },
    });
    if (values.help) {
      console.log(USAGE);
      return 0;
    }
    check = values.check;
  } catch (err) {
    console.error(`${err instanceof Error ? err.message : String(err)}\n${USAGE}`);
    return 2;
  }
  return runReport({ resultsDir: 'results', readmePath: 'README.md', check });
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
