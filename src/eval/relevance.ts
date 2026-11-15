// Relevance labels (docs/PLAN.md §6): a retrieved chunk covers an evidence span if its source_path equals the
// span's source and its normalized content contains the normalized span.
import { normalize } from './normalize';

export interface RetrievedChunk {
  sourcePath: string;
  content: string;
}

export interface EvidenceSpan {
  source: string;
  span: string;
}

/**
 * Per chunk, in rank order, the indices of the evidence spans it covers. Chunks are normalized with
 * `fromLineStart: false`: a chunk can start mid-line, and its first line must normalize as it does inside the body
 * the validator checked the span against (a leading `2. ` stays text there, it is no list marker).
 */
export function coverage(chunks: readonly RetrievedChunk[], evidence: readonly EvidenceSpan[]): number[][] {
  const spans = evidence.map((e) => ({ source: e.source, span: normalize(e.span) }));
  return chunks.map((chunk) => {
    const relevant = spans.flatMap((s, i) => (s.source === chunk.sourcePath ? [i] : []));
    if (relevant.length === 0) {
      return [];
    }
    const text = normalize(chunk.content, { fromLineStart: false });
    return relevant.filter((i) => text.includes(spans[i]?.span ?? '\0'));
  });
}
