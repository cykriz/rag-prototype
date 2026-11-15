// Golden dataset (docs/PLAN.md M10, §6): Zod schema of eval/golden.jsonl, parser, span check against the loaded
// document bodies, count checks against the §6 targets, and the dataset hash stored in every result file.
import { readFile } from 'node:fs/promises';
import { z } from 'zod';
import { sha256 } from '../ingest/hash';
import { loadMarkdown, MARKDOWN_FILE } from '../ingest/loadMarkdown';
import { normalize } from './normalize';

export const DEFAULT_DATASET = 'eval/golden.jsonl';

export const SPLITS = ['dev', 'test'] as const;
export const CATEGORIES = [
  'direct_fact',
  'how_to',
  'multi_document',
  'vocabulary_mismatch',
  'negative',
] as const;
export type Category = (typeof CATEGORIES)[number];

/** Entries per category (§6); 45 in total. */
export const CATEGORY_TARGETS: Record<Category, number> = {
  direct_fact: 15,
  how_to: 10,
  multi_document: 8,
  vocabulary_mismatch: 5,
  negative: 7,
};
export const TOTAL_TARGET = Object.values(CATEGORY_TARGETS).reduce((a, b) => a + b, 0);
export const DEV_TARGET = 15;
export const MAX_SPAN_WORDS = 15;

/** Dev entries a category of `n` should have for a 15/45 split stratified by category (5/3/3/2/2 at the targets). */
export function devTarget(n: number): number {
  return Math.round((n * DEV_TARGET) / TOTAL_TARGET);
}

function sourceProblem(source: string): string | undefined {
  if (source.startsWith('/') || /^[A-Za-z]:/.test(source)) {
    return 'must be relative to the docs root, like documents.source_path';
  }
  if (source.startsWith('docs/')) {
    return 'drop the "docs/" prefix: paths are relative to the docs root, like documents.source_path';
  }
  if (source.includes('\\') || source.split('/').some((s) => s === '..' || s === '.' || s === '')) {
    return 'must be a plain relative path with "/" separators';
  }
  if (!MARKDOWN_FILE.test(source)) {
    return 'must be a .md or .mdx file';
  }
  return undefined;
}

const Source = z.string().superRefine((source, ctx) => {
  const problem = sourceProblem(source);
  if (problem) {
    ctx.addIssue({ code: 'custom', message: problem });
  }
});

const Evidence = z.strictObject({ source: Source, span: z.string().min(1) });

function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

export const GoldenEntry = z
  .strictObject({
    id: z.string().regex(/^q\d{3}$/, 'expected q followed by three digits, like q007'),
    split: z.enum(SPLITS),
    category: z.enum(CATEGORIES),
    question: z.string().trim().min(1),
    answerable: z.boolean(),
    evidence: z.array(Evidence),
    keywords: z.array(z.string().trim().min(1)),
    reference_answer: z.string().trim().min(1),
    origin: z.literal('llm').optional(),
  })
  .superRefine((e, ctx) => {
    const issue = (path: (string | number)[], message: string) =>
      ctx.addIssue({ code: 'custom', path, message });
    const negative = e.category === 'negative';
    if (negative === e.answerable) {
      issue(['answerable'], 'must be false for category negative and true for every other category');
    }
    if (!e.answerable && e.evidence.length > 0) {
      issue(['evidence'], 'must be empty for an unanswerable entry');
    }
    if (e.answerable && e.evidence.length === 0) {
      issue(['evidence'], 'an answerable entry needs at least one span');
    }
    if (e.answerable && e.keywords.length === 0) {
      issue(['keywords'], 'an answerable entry needs at least one keyword (keyword coverage divides by it)');
    }
    if (e.category === 'multi_document' && new Set(e.evidence.map((ev) => ev.source)).size < 2) {
      issue(['evidence'], 'multi_document needs spans from at least 2 distinct sources');
    }
    const seen = new Set<string>();
    e.evidence.forEach((ev, i) => {
      // Count words after normalization, so table pipes and other markup tokens are not words.
      const words = wordCount(normalize(ev.span));
      if (words > MAX_SPAN_WORDS) {
        issue(['evidence', i, 'span'], `${words} words, at most ${MAX_SPAN_WORDS}`);
      }
      const norm = normalize(ev.span);
      if (!norm) {
        issue(['evidence', i, 'span'], 'empty after normalization');
      }
      const key = `${ev.source}\0${norm}`;
      if (seen.has(key)) {
        issue(['evidence', i, 'span'], 'same source and span as an earlier evidence entry');
      }
      seen.add(key);
    });
  });
export type GoldenEntry = z.infer<typeof GoldenEntry>;

export type IssueKind =
  | 'invalid_json'
  | 'schema'
  | 'duplicate_id'
  | 'source_not_found'
  | 'span_not_found'
  | 'count_total'
  | 'count_split'
  | 'count_category'
  | 'split_category'
  | 'origin_llm';

/** Kinds that are warnings while the dataset is incomplete and errors under --strict. */
export const COUNT_KINDS: ReadonlySet<IssueKind> = new Set([
  'count_total',
  'count_split',
  'count_category',
  'split_category',
]);

export interface Issue {
  level: 'error' | 'warning';
  /** Entry id, `line N` when the id is unknown, `-` for dataset-level findings. */
  ref: string;
  /** Evidence source, `-` when the finding is not about one. */
  source: string;
  kind: IssueKind;
  detail: string;
}

export interface ParsedDataset {
  /** Non-blank lines. */
  lines: number;
  /** Entries that passed the schema, first occurrence of each id only. */
  entries: GoldenEntry[];
  issues: Issue[];
}

function evidenceSource(raw: unknown, path: PropertyKey[]): string {
  if (path[0] !== 'evidence' || typeof path[1] !== 'number' || typeof raw !== 'object' || raw === null) {
    return '-';
  }
  const ev = (raw as { evidence?: unknown }).evidence;
  const item = Array.isArray(ev) ? (ev[path[1]] as { source?: unknown } | undefined) : undefined;
  return typeof item?.source === 'string' ? item.source : '-';
}

/** Pure: JSON per line, schema and cross-field rules, unique ids. Blank lines are skipped. */
export function parseDataset(text: string): ParsedDataset {
  const entries: GoldenEntry[] = [];
  const issues: Issue[] = [];
  const idLines = new Map<string, number>();
  let lines = 0;
  text.split(/\r?\n/).forEach((line, i) => {
    if (!line.trim()) {
      return;
    }
    lines++;
    const lineRef = `line ${i + 1}`;
    let raw: unknown;
    try {
      raw = JSON.parse(line);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      issues.push({ level: 'error', ref: lineRef, source: '-', kind: 'invalid_json', detail });
      return;
    }
    const rawId = typeof raw === 'object' && raw !== null ? (raw as { id?: unknown }).id : undefined;
    const ref = typeof rawId === 'string' && rawId ? rawId : lineRef;
    // Ids are tracked before the schema check, so a repeat is reported even if the first line is invalid.
    if (typeof rawId === 'string' && rawId) {
      const first = idLines.get(rawId);
      if (first !== undefined) {
        const detail = `${lineRef} repeats the id of line ${first}`;
        issues.push({ level: 'error', ref, source: '-', kind: 'duplicate_id', detail });
        return;
      }
      idLines.set(rawId, i + 1);
    }
    const parsed = GoldenEntry.safeParse(raw);
    if (!parsed.success) {
      for (const zi of parsed.error.issues) {
        const where = zi.path.length ? `${zi.path.join('.')}: ` : '';
        const source = evidenceSource(raw, zi.path);
        issues.push({ level: 'error', ref, source, kind: 'schema', detail: `${where}${zi.message}` });
      }
      return;
    }
    entries.push(parsed.data);
  });
  return { lines, entries, issues };
}

/** Document body as chunked by ingest (loadMarkdown), or undefined when the source file does not exist. */
export type BodyReader = (source: string) => Promise<string | undefined>;

/**
 * loadMarkdown body, as ingest chunks it. Undefined unless `source` is exactly one of `files` (listCorpus, the paths
 * ingest stores as documents.source_path), so a wrong-case path fails on case-insensitive file systems too.
 */
export function corpusReader(root: string, files: readonly string[]): BodyReader {
  const known = new Set(files);
  return async (source) => (known.has(source) ? (await loadMarkdown(root, source)).body : undefined);
}

/** Every span must occur in the normalized body of its source, the text chunks are cut from. */
export async function checkSpans(entries: GoldenEntry[], readBody: BodyReader): Promise<Issue[]> {
  const bodies = new Map<string, string | undefined>();
  const issues: Issue[] = [];
  for (const e of entries) {
    const missing = new Set<string>();
    for (const { source, span } of e.evidence) {
      if (!bodies.has(source)) {
        const body = await readBody(source);
        bodies.set(source, body === undefined ? undefined : normalize(body));
      }
      const body = bodies.get(source);
      if (body === undefined) {
        if (!missing.has(source)) {
          missing.add(source);
          issues.push({
            level: 'error',
            ref: e.id,
            source,
            kind: 'source_not_found',
            detail: 'no such file',
          });
        }
      } else if (!body.includes(normalize(span))) {
        issues.push({
          level: 'error',
          ref: e.id,
          source,
          kind: 'span_not_found',
          detail: JSON.stringify(span),
        });
      }
    }
  }
  return issues;
}

/** Warnings against the §6 targets: total, split, category counts, stratified split, LLM-written entries. */
export function countWarnings(entries: GoldenEntry[]): Issue[] {
  const warn = (kind: IssueKind, detail: string): Issue => ({
    level: 'warning',
    ref: '-',
    source: '-',
    kind,
    detail,
  });
  const issues: Issue[] = [];
  if (entries.length !== TOTAL_TARGET) {
    issues.push(warn('count_total', `${entries.length}/${TOTAL_TARGET} entries`));
  }
  const dev = entries.filter((e) => e.split === 'dev').length;
  const test = entries.length - dev;
  const testTarget = TOTAL_TARGET - DEV_TARGET;
  if (dev !== DEV_TARGET || test !== testTarget) {
    issues.push(warn('count_split', `dev ${dev}/${DEV_TARGET}, test ${test}/${testTarget}`));
  }
  for (const category of CATEGORIES) {
    const inCategory = entries.filter((e) => e.category === category);
    const n = inCategory.length;
    if (n !== CATEGORY_TARGETS[category]) {
      issues.push(warn('count_category', `${category} ${n}/${CATEGORY_TARGETS[category]}`));
    }
    const catDev = inCategory.filter((e) => e.split === 'dev').length;
    if (n > 0 && catDev !== devTarget(n)) {
      issues.push(warn('split_category', `${category}: dev ${catDev} of ${n}, expected ${devTarget(n)}`));
    }
  }
  const llm = entries.filter((e) => e.origin === 'llm').map((e) => e.id);
  if (llm.length > 0) {
    issues.push(
      warn('origin_llm', `${llm.length} entries written by an LLM, reported separately: ${llm.join(', ')}`),
    );
  }
  return issues;
}

export interface LoadedDataset {
  text: string;
  /** sha256 of the raw file bytes; stored in every result file (M11+). */
  hash: string;
}

/** The dataset file and its hash, or undefined when it does not exist. */
export async function loadDataset(path: string): Promise<LoadedDataset | undefined> {
  let bytes: Uint8Array;
  try {
    bytes = await readFile(path);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      return undefined;
    }
    throw err;
  }
  return { text: new TextDecoder().decode(bytes), hash: sha256(bytes) };
}
