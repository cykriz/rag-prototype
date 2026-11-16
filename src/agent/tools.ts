// The agent's search_docs tool (docs/PLAN.md M18a): definition, argument parsing and the tool message content.
import type { Document } from '@langchain/core/documents';
import { z } from 'zod';
import type { ToolCall, ToolDef } from '../llm/types';
import { MAX_TOOL_K } from '../rag/pipeline';
import { formatContext } from '../rag/prompts';
import { SearchQueryText } from '../rag/schemas';
import type { ChunkMetadata } from '../retrieval/pgRetriever';

export const SEARCH_DOCS = 'search_docs';

/** The tool definition; `defaultK` (the request's --k) is what an absent k means. */
export function searchDocsTool(defaultK: number): ToolDef {
  return {
    name: SEARCH_DOCS,
    description:
      'Search the Bun documentation. Returns the most relevant chunks, each starting with a [chunk:<id>] header.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query in English' },
        k: {
          type: 'integer',
          minimum: 1,
          maximum: MAX_TOOL_K,
          description: `Number of chunks to return (default ${defaultK})`,
        },
      },
      required: ['query'],
    },
  };
}

/** Integer strings ("5") count as numbers; models sometimes quote them. */
const IntLike = z.preprocess(
  (v) => (typeof v === 'string' && /^\s*-?\d+\s*$/.test(v) ? Number(v) : v),
  z.number().int().nullish(),
);

const SearchArgs = z.object({ query: SearchQueryText, k: IntLike });

export type SearchArgs =
  | {
      ok: true;
      query: string;
      /** k as the model sent it; null when absent. */
      kRequested: number | null;
      /** k used: kRequested clamped to 1–MAX_TOOL_K, or defaultK. */
      k: number;
    }
  | { ok: false; error: string };

/**
 * Validates one tool call. An unknown tool, unparsable JSON, a blank or over-long query and a non-integer k are
 * errors, reported back to the model as the tool result; an integer k out of range is clamped, not an error.
 */
export function parseSearchArgs(call: Pick<ToolCall, 'name' | 'arguments'>, defaultK: number): SearchArgs {
  if (call.name !== SEARCH_DOCS) {
    return { ok: false, error: `unknown tool "${call.name}"; the only tool is ${SEARCH_DOCS}` };
  }
  let json: unknown;
  try {
    json = JSON.parse(call.arguments);
  } catch {
    return { ok: false, error: 'arguments are not valid JSON' };
  }
  const parsed = SearchArgs.safeParse(json);
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `${i.path.join('.') || 'arguments'}: ${i.message}`);
    return { ok: false, error: `invalid arguments (${issues.join('; ')})` };
  }
  const kRequested = parsed.data.k ?? null;
  const k = kRequested === null ? defaultK : Math.min(Math.max(kRequested, 1), MAX_TOOL_K);
  return { ok: true, query: parsed.data.query, kRequested, k };
}

/** The tool message for the model: the chunks as in the answer prompt, or a note when nothing matched. */
export const searchResultContent = (docs: Document<ChunkMetadata>[]): string =>
  docs.length > 0 ? formatContext(docs) : 'No results.';

export const toolErrorContent = (error: string): string => `error: ${error}`;
