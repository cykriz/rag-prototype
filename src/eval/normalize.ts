// Text normalization for evidence spans (docs/PLAN.md §6 relevance): lowercase, whitespace collapsed, Markdown
// markup removed. The golden validator (dataset.ts) and the relevance check (relevance.ts) both use this function,
// on the span and on the text, so a span that validates against a document body also matches its chunks.

// Images and inline links keep their text: `![alt](url)` and `[text](url)` become `alt` and `text`.
const LINK = /!?\[([^\]\n]*)\]\([^)\n]*\)/g;
const AUTOLINK = /<((?:https?|mailto):[^>\s]+)>/g;
// Fence delimiter lines, with their info string.
const FENCE_LINE = /^[ \t]*(?:`{3,}|~{3,})[^\n]*$/gm;
// Block markers at a line start, nested ones included (`> - item`): headings, quotes, list items.
const LINE_MARKERS = /^[ \t]*(?:(?:#{1,6}|>|[-*+]|\d+[.)])(?:[ \t]+|$))+/gm;
// Backslash escapes of ASCII punctuation.
const ESCAPE = /\\([!-/:-@[-`{-~])/g;
// Code, emphasis, strikethrough and table markup.
const INLINE_MARKUP = /[`*|]|~~/g;
// `_` only as an emphasis delimiter, never inside a word (`NODE_ENV`, `snake_case`).
const UNDERSCORE_EMPHASIS = /(?<![\p{L}\p{N}])_+|_+(?![\p{L}\p{N}])/gu;

export interface NormalizeOptions {
  /**
   * Whether `text` starts at a line start (default). False for a chunk, which may be cut mid-line: its first line
   * then keeps a leading block marker or fence-like run, as the same text does inside the full body (relevance.ts).
   */
  fromLineStart?: boolean;
}

/**
 * Lowercase, Markdown markup removed, whitespace collapsed to single spaces, trimmed. Block markers count only at
 * a line start, so a span must not cross a line break that starts a list item, heading or quote.
 */
export function normalize(text: string, { fromLineStart = true }: NormalizeOptions = {}): string {
  // The offset is the first number among the replacer arguments (capture groups are strings or undefined), so a
  // capture group added to FENCE_LINE or LINE_MARKERS later cannot shift it.
  const atLineStart = (match: string, ...rest: unknown[]) =>
    fromLineStart || rest.find((x) => typeof x === 'number') !== 0 ? '' : match;
  return text
    .replace(/\r\n?/g, '\n')
    .replace(LINK, '$1')
    .replace(AUTOLINK, '$1')
    .replace(FENCE_LINE, atLineStart)
    .replace(LINE_MARKERS, atLineStart)
    .replace(ESCAPE, '$1')
    .replace(INLINE_MARKUP, '')
    .replace(UNDERSCORE_EMPHASIS, '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}
