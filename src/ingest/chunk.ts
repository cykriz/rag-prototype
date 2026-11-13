// Markdown chunker (docs/PLAN.md M6): RecursiveCharacterTextSplitter (markdown) with a cl100k token length
// function, fences that fit kept whole, hard max of chunkSize tokens, heading path per chunk.
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { countTokens } from '../llm/tokens';
import { findFences, findHeadings, type Heading } from './fences';

export interface ChunkConfig {
  key: string;
  chunkSize: number;
  chunkOverlap: number;
  splitter: 'markdown';
}

export interface Chunk {
  ordinal: number;
  /** Heading texts from the document title down, joined with " > ". */
  headingPath: string;
  content: string;
  tokenCount: number;
}

/**
 * Parses `md-<size>-<overlap>`. Value ranges are enforced by the chunk_configs CHECKs (migrations/001_init.sql)
 * at insert, and overlap >= size by the splitter itself.
 */
export function parseChunkConfigKey(key: string): ChunkConfig {
  const m = /^md-([1-9]\d*)-(\d+)$/.exec(key);
  if (!m) {
    throw new Error(`Invalid chunk config key "${key}", expected md-<size>-<overlap>`);
  }
  return { key, chunkSize: Number(m[1]), chunkOverlap: Number(m[2]), splitter: 'markdown' };
}

// Private-use characters with no whitespace: the splitter only cuts such a run at character level, which it never
// reaches for a protected fence (see protect()).
const OPEN = '\uE000';
const CLOSE = '\uE001';
const SENTINEL = /\uE000(\d+)\uE001/g;
const HEADING_SEPARATOR = ' > ';

interface Piece {
  start: number;
  text: string;
}

/**
 * Replaces each fence that fits by a sentinel. The splitter keeps a piece whole only if its length is strictly
 * below chunkSize, and a piece carries its leading separator, hence the "\n\n" and the strict comparison.
 */
function protect(body: string, chunkSize: number): { text: string; fences: string[] } {
  const fences: string[] = [];
  let text = '';
  let last = 0;
  for (const f of findFences(body)) {
    const fence = body.slice(f.start, f.end);
    if (countTokens(`\n\n${fence}`) < chunkSize) {
      text += `${body.slice(last, f.start)}${OPEN}${fences.length}${CLOSE}`;
      fences.push(fence);
      last = f.end;
    }
  }
  return { text: text + body.slice(last), fences };
}

/** Trims a slice of `text` and returns it with its absolute start offset, or nothing if it is blank. */
function trimmed(text: string, from: number, to: number): Piece[] {
  const raw = text.slice(from, to);
  const t = raw.trim();
  return t ? [{ start: from + raw.length - raw.trimStart().length, text: t }] : [];
}

/** Halves an oversized piece at the newline nearest its middle (character cut for a single line), recursively. */
function enforceMax(piece: Piece, len: (s: string) => number, max: number): Piece[] {
  const { text } = piece;
  // A lone sentinel cannot be cut (the character cut below would move past it and recurse forever). protect()
  // keeps such pieces within max, so this branch is a termination guard; the final token check still applies.
  const loneSentinel = text.startsWith(OPEN) && text.endsWith(CLOSE) && !text.includes('\n');
  if (len(text) <= max || loneSentinel) {
    return [piece];
  }
  const mid = text.length / 2;
  let cut = -1;
  for (let i = text.indexOf('\n'); i > 0; i = text.indexOf('\n', i + 1)) {
    if (cut < 0 || Math.abs(i - mid) < Math.abs(cut - mid)) {
      cut = i;
    }
  }
  if (cut < 0) {
    cut = Math.floor(mid);
    const open = text.lastIndexOf(OPEN, cut - 1);
    if (open >= 0 && text.indexOf(CLOSE, open) >= cut) {
      cut = open > 0 ? open : text.indexOf(CLOSE, open) + 1;
    }
    if (/[\uDC00-\uDFFF]/.test(text[cut] ?? '')) {
      cut -= 1;
    }
  }
  return [...trimmed(text, 0, cut), ...trimmed(text, cut, text.length)].flatMap((p) =>
    enforceMax({ start: piece.start + p.start, text: p.text }, len, max),
  );
}

function headingPathAt(headings: Heading[], title: string, offset: number): string {
  const stack: { level: number; text: string }[] = title ? [{ level: 1, text: title }] : [];
  for (const h of headings) {
    if (h.offset > offset) {
      break;
    }
    while (stack.length > 0 && (stack.at(-1)?.level ?? 0) >= h.level) {
      stack.pop();
    }
    stack.push(h);
  }
  return stack.map((s) => s.text).join(HEADING_SEPARATOR);
}

export async function chunkMarkdown(body: string, title: string, config: ChunkConfig): Promise<Chunk[]> {
  if (body.includes(OPEN) || body.includes(CLOSE)) {
    throw new Error('Input contains the private-use characters U+E000/U+E001 reserved for fence protection');
  }
  if (body.trim() === '') {
    return [];
  }
  const { chunkSize, chunkOverlap } = config;
  const { text, fences } = protect(body, chunkSize);
  const restore = (s: string) =>
    s.replace(SENTINEL, (_, i: string) => {
      const fence = fences[Number(i)];
      if (fence === undefined) {
        throw new Error(`Unknown fence sentinel ${i}`);
      }
      return fence;
    });
  const len = (s: string) => countTokens(restore(s));

  const splitter = RecursiveCharacterTextSplitter.fromLanguage('markdown', {
    chunkSize,
    chunkOverlap,
    lengthFunction: len,
  });
  const split = await splitter.splitText(text);

  // Each chunk is a trimmed contiguous substring of `text`; locate it to know its heading context. It starts after
  // the previous chunk's start and, overlap or not, no later than the previous end plus whitespace.
  const located: Piece[] = [];
  let prev: Piece | undefined;
  for (const s of split) {
    const from = prev ? prev.start + 1 : 0;
    const start = text.indexOf(s, from);
    let latest = text.length;
    if (prev) {
      latest = prev.start + prev.text.length;
      while (latest < text.length && /\s/.test(text[latest] ?? '')) {
        latest += 1;
      }
    }
    if (start < 0 || start > latest) {
      throw new Error('Splitter returned a chunk that is not at its expected position in the input');
    }
    prev = { start, text: s };
    located.push(prev);
  }

  const headings = findHeadings(text);
  return located
    .flatMap((p) => enforceMax(p, len, chunkSize))
    .map((p, ordinal) => {
      const content = restore(p.text);
      if (content.includes(OPEN) || content.includes(CLOSE)) {
        throw new Error('A fence sentinel was cut; protected fences must stay whole');
      }
      const tokenCount = countTokens(content);
      if (tokenCount > chunkSize) {
        throw new Error(`Chunk ${ordinal} has ${tokenCount} tokens, above chunk size ${chunkSize}`);
      }
      return { ordinal, headingPath: headingPathAt(headings, title, p.start), content, tokenCount };
    });
}
