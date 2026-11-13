// Markdown/MDX loader (docs/PLAN.md M6): frontmatter removal, MDX stripping, title extraction.
import { readdir, readFile } from 'node:fs/promises';
import { basename, extname, join, relative, sep } from 'node:path';
import { findFences, findHeadings } from './fences';

export interface ParsedMarkdown {
  title: string;
  body: string;
}

export interface LoadedDocument extends ParsedMarkdown {
  /** Relative to the corpus root, always with `/`. */
  sourcePath: string;
}

const FRONTMATTER = /^---\n([\s\S]*?)\n---[ \t]*(?:\n|$)/;
const FM_TITLE = /^title:[ \t]*(.+?)[ \t]*$/m;
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
  const fm = FRONTMATTER.exec(text);
  if (fm) {
    const t = FM_TITLE.exec(fm[1] ?? '')?.[1];
    fmTitle = t === undefined ? undefined : unquote(t);
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
  return { title: fmTitle || h1 || fallbackTitle, body };
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
    .filter((e) => e.isFile() && /\.mdx?$/i.test(e.name))
    .map((e) => relative(root, join(e.parentPath, e.name)).split(sep).join('/'))
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}
