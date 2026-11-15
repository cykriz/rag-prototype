// bun run eval:validate [--file eval/golden.jsonl] [--root corpus/bun-docs/docs] [--strict]
// Validates the golden dataset (docs/PLAN.md M10): schema and cross-field rules, every evidence span present in
// its source after normalization, counts against the §6 targets. Prints one line per finding, then a summary with
// the dataset sha256. Exit 0 without errors, 1 with errors or a missing dataset/corpus, 2 on a usage error.
import { parseArgs } from 'node:util';
import {
  COUNT_KINDS,
  checkSpans,
  corpusReader,
  countWarnings,
  DEFAULT_DATASET,
  DEV_TARGET,
  type Issue,
  loadDataset,
  parseDataset,
  TOTAL_TARGET,
} from '../eval/dataset';
import { DEFAULT_CORPUS_ROOT, listCorpus } from '../ingest/loadMarkdown';

const USAGE =
  `Usage: bun run eval:validate [--file ${DEFAULT_DATASET}] [--root ${DEFAULT_CORPUS_ROOT}] [--strict]\n` +
  `  --strict  count deviations from the ${TOTAL_TARGET} entries / ${DEV_TARGET} dev / ${TOTAL_TARGET - DEV_TARGET} test ` +
  'targets as errors (M10 acceptance)';

function parse(argv: string[]) {
  const { values } = parseArgs({
    args: argv,
    strict: true,
    allowPositionals: false,
    options: {
      file: { type: 'string', default: DEFAULT_DATASET },
      root: { type: 'string', default: DEFAULT_CORPUS_ROOT },
      strict: { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  return values;
}

function formatIssue(i: Issue): string {
  return [i.level === 'error' ? 'ERROR' : 'WARN ', i.ref, i.source, i.kind, i.detail].join('  ');
}

function plural(n: number, word: string): string {
  return `${n} ${word}${n === 1 ? '' : 's'}`;
}

export async function main(argv: string[]): Promise<number> {
  let args: ReturnType<typeof parse>;
  try {
    args = parse(argv);
  } catch (err) {
    console.error(`${err instanceof Error ? err.message : String(err)}\n${USAGE}`);
    return 2;
  }
  if (args.help) {
    console.log(USAGE);
    return 0;
  }

  try {
    const dataset = await loadDataset(args.file);
    if (!dataset?.text.trim()) {
      console.error(`${args.file} missing or empty: write the questions per docs/PLAN.md §6`);
      return 1;
    }
    const { lines, entries, issues } = parseDataset(dataset.text);
    // A missing or empty corpus is reported after the findings; the schema checks above still count.
    // Only the listing is caught here; a read error during the span check reaches the outer handler.
    let corpusError: string | undefined;
    let files: string[] | undefined;
    try {
      files = await listCorpus(args.root);
    } catch (err) {
      corpusError = err instanceof Error ? err.message : String(err);
    }
    if (files) {
      issues.push(...(await checkSpans(entries, corpusReader(args.root, files))));
    }
    for (const w of countWarnings(entries)) {
      issues.push(args.strict && COUNT_KINDS.has(w.kind) ? { ...w, level: 'error' } : w);
    }

    for (const i of issues) {
      console.log(formatIssue(i));
    }
    if (corpusError) {
      console.error(`${corpusError} (spans not checked)`);
    }
    const errors = issues.filter((i) => i.level === 'error').length;
    const warnings = issues.length - errors;
    const dev = entries.filter((e) => e.split === 'dev').length;
    console.log(
      `${lines} ${lines === 1 ? 'entry' : 'entries'}, ${entries.length} valid ` +
        `(${dev} dev / ${entries.length - dev} test): ${plural(errors, 'error')}, ${plural(warnings, 'warning')}; ` +
        `sha256 ${dataset.hash}`,
    );
    return errors > 0 || corpusError ? 1 : 0;
  } catch (err) {
    console.error(err instanceof Error ? err.message : String(err));
    return 1;
  }
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
