// Shared Markdown block scanner (docs/PLAN.md M6): the one place that decides what is a code fence and what is a
// heading, so MDX stripping, fence protection in the chunker and heading paths cannot disagree.

export interface Range {
  /** Offset of the opening fence line's first character. */
  start: number;
  /** Offset just past the closing fence line (its newline excluded), or the text length if unclosed. */
  end: number;
}

export interface Heading {
  /** Offset of the first `#`. */
  offset: number;
  level: number;
  text: string;
}

// Any indent: MDX component bodies (e.g. <Step>) indent their fences. Backtick info strings contain no backtick.
const OPEN = /^[ \t]*(`{3,}(?=[^`]*$)|~{3,})/;
const CLOSE = /^[ \t]*(`{3,}|~{3,})[ \t]*$/;
const HEADING = /^( {0,3})(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$/;

interface Line {
  start: number;
  text: string;
}

function lines(text: string): Line[] {
  const out: Line[] = [];
  let start = 0;
  for (const line of text.split('\n')) {
    out.push({ start, text: line });
    start += line.length + 1;
  }
  return out;
}

/** Fenced code blocks in document order. A closing fence uses the opening character at no shorter length. */
export function findFences(text: string): Range[] {
  const fences: Range[] = [];
  let open: { start: number; marker: string } | undefined;
  for (const line of lines(text)) {
    if (open) {
      const close = CLOSE.exec(line.text)?.[1];
      if (close && close[0] === open.marker[0] && close.length >= open.marker.length) {
        fences.push({ start: open.start, end: line.start + line.text.length });
        open = undefined;
      }
      continue;
    }
    const marker = OPEN.exec(line.text)?.[1];
    if (marker) {
      open = { start: line.start, marker };
    }
  }
  if (open) {
    fences.push({ start: open.start, end: text.length });
  }
  return fences;
}

const insideAny = (ranges: Range[], offset: number): boolean =>
  ranges.some((r) => offset >= r.start && offset < r.end);

/** ATX headings outside fences, trailing `#`s removed, text otherwise verbatim. */
export function findHeadings(text: string): Heading[] {
  const fences = findFences(text);
  const headings: Heading[] = [];
  for (const line of lines(text)) {
    const m = HEADING.exec(line.text);
    if (m && !insideAny(fences, line.start)) {
      const [, indent = '', hashes = '', title = ''] = m;
      headings.push({ offset: line.start + indent.length, level: hashes.length, text: title });
    }
  }
  return headings;
}
