// Markdown/MDX loader (docs/PLAN.md M6): frontmatter removal, MDX stripping, title extraction.
import { readdir, readFile } from 'node:fs/promises';
import { basename, extname, join, relative, sep } from 'node:path';
import { findFences, findHeadings } from './fences';

export interface ParsedMarkdown {
  title: string;
  body: string;
  /** Frontmatter `description` (M19b); absent when missing or empty. Indexed only by `mdfm-*` chunk configs. */
  description?: string;
}

export interface LoadedDocument extends ParsedMarkdown {
  /** Relative to the corpus root, always with `/`. */
  sourcePath: string;
}

/** Where scripts/fetch-corpus.sh puts the Bun docs; documents.source_path and evidence sources are relative to it. */
export const DEFAULT_CORPUS_ROOT = 'corpus/bun-docs/docs';

const FETCH_HINT = 'run ./scripts/fetch-corpus.sh';

/** File names the corpus loader picks up (listMarkdownFiles); evidence sources must match it too. */
export const MARKDOWN_FILE = /\.mdx?$/i;

const FRONTMATTER = /^---\n([\s\S]*?)\n---[ \t]*(?:\n|$)/;
const FM_TITLE = /^title:[ \t]*(.+?)[ \t]*$/m;
const FM_DESCRIPTION = /^description:[ \t]*(.*?)[ \t]*$/m;
const MDX_COMMENT = /\{\/\*[\s\S]*?\*\/\}/g;
// MDX ESM lines only (import ... from '...', side-effect imports, export declarations), never prose.
const MDX_ESM =
  /^(?:import\s[^\n]*?\sfrom\s+['"][^'"\n]+['"];?|import\s+['"][^'"\n]+['"];?|export\s+(?:const|let|function|default)\b[^\n]*)[ \t]*$/gm;
// Capitalised JSX components, opening, closing or self-closing; attributes may span lines and hold quoted strings
// (which may contain `<`, `>` or backticks) and `{...}` expressions.
const JSX_TAG = /<\/?[A-Z][\w.]*(?:\s(?:"[^"]*"|'[^']*'|\{[^{}]*\}|[^<>{}"'])*)?\/?>/g;
const MDX_SYNTAX = new RegExp(`${MDX_COMMENT.source}|${MDX_ESM.source}|${JSX_TAG.source}`, 'gm');
// Inline code spans, matched by equal backtick runs and never across a blank line (a stray backtick must not
// shield later paragraphs); MDX syntax starting inside one is left untouched.
const INLINE_CODE = /(`+)(?:(?!\n[ \t]*\n)[\s\S])*?[^`]\1(?!`)/g;

function unquote(value: string): string {
  const q = value[0];
  return (q === '"' || q === "'") && value.endsWith(q) && value.length >= 2 ? value.slice(1, -1) : value;
}

/**
 * A single-line YAML scalar: quotes removed, plus the \" and \\ escapes in a double-quoted one and '' in a
 * single-quoted one. A block scalar
 * indicator (`|`, `>`) yields '' (not used by the corpus frontmatter).
 */
function yamlScalar(value: string): string {
  if (/^[|>][-+]?$/.test(value)) {
    return '';
  }
  const v = unquote(value);
  if (v === value) {
    return v;
  }
  return value.startsWith('"') ? v.replace(/\\(["\\])/g, '$1') : v.replaceAll("''", "'");
}

function stripMdx(prose: string): string {
  const code = [...prose.matchAll(INLINE_CODE)].map((m) => ({ start: m.index, end: m.index + m[0].length }));
  let out = '';
  let last = 0;
  for (const m of prose.matchAll(MDX_SYNTAX)) {
    if (!code.some((c) => m.index >= c.start && m.index < c.end)) {
      out += prose.slice(last, m.index);
      last = m.index + m[0].length;
    }
  }
  out += prose.slice(last);
  // Lines emptied by stripping count as blank; then at most one blank line in a row.
  return out
    .split('\n')
    .map((l) => (l.trim() === '' ? '' : l))
    .join('\n')
    .replace(/\n{3,}/g, '\n\n');
}

/** Pure: raw file content in, clean Markdown body and title out. Fences are never modified. */
export function parseMarkdown(raw: string, fallbackTitle: string): ParsedMarkdown {
  let text = raw.replace(/\r\n?/g, '\n');
  let fmTitle: string | undefined;
  let description: string | undefined;
  const fm = FRONTMATTER.exec(text);
  if (fm) {
    const t = FM_TITLE.exec(fm[1] ?? '')?.[1];
    fmTitle = t === undefined ? undefined : unquote(t);
    const d = FM_DESCRIPTION.exec(fm[1] ?? '')?.[1];
    description = d === undefined ? undefined : yamlScalar(d).trim() || undefined;
    text = text.slice(fm[0].length);
  }

  let body = '';
  let last = 0;
  for (const f of findFences(text)) {
    body += stripMdx(text.slice(last, f.start)) + text.slice(f.start, f.end);
    last = f.end;
  }
  // No blank-line collapsing here: it would also hit fences. stripMdx already collapses prose.
  body = (body + stripMdx(text.slice(last))).trim();

  const h1 = findHeadings(body).find((h) => h.level === 1)?.text;
  const title = fmTitle || h1 || fallbackTitle;
  return description === undefined ? { title, body } : { title, body, description };
}

export async function loadMarkdown(root: string, sourcePath: string): Promise<LoadedDocument> {
  const raw = await readFile(join(root, sourcePath), 'utf8');
  const stem = basename(sourcePath, extname(sourcePath));
  return { sourcePath, ...parseMarkdown(raw, stem) };
}

/** All .md/.mdx files below root as relative `/` paths, sorted by code unit for a stable order. */
export async function listMarkdownFiles(root: string): Promise<string[]> {
  const entries = await readdir(root, { recursive: true, withFileTypes: true });
  return entries
    .filter((e) => e.isFile() && MARKDOWN_FILE.test(e.name))
    .map((e) => relative(root, join(e.parentPath, e.name)).split(sep).join('/'))
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

/** listMarkdownFiles, but a missing or empty corpus throws with the fetch hint. */
export async function listCorpus(root: string): Promise<string[]> {
  let files: string[];
  try {
    files = await listMarkdownFiles(root);
  } catch (err) {
    const code = (err as NodeJS.ErrnoException).code;
    if (code === 'ENOENT' || code === 'ENOTDIR') {
      throw new Error(`Corpus directory ${root} not found; ${FETCH_HINT}`, { cause: err });
    }
    throw err;
  }
  if (files.length === 0) {
    throw new Error(`No .md/.mdx files under ${root}; ${FETCH_HINT}`);
  }
  return files;
}
