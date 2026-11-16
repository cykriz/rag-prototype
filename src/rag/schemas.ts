// Zod schemas for structured model output (docs/PLAN.md M9 answer, M15 rewrite, M16 expand, M17 rerank).
import { z } from 'zod';
import { MAX_EXPANSIONS, MAX_RERANK_SCORE } from './pipeline';
import { NOT_FOUND } from './prompts';

export const Answer = z.object({
  found: z.boolean(),
  answer: z.string(),
  citations: z.array(z.number().int()),
});
export type Answer = z.infer<typeof Answer>;

/** An abstention, however `found` is set; normalized to found=false by answer.ts. */
export const isAbstention = (a: Answer): boolean => !a.found || a.answer.trim() === NOT_FOUND;

/**
 * Answer, refined against the chunk IDs of one request's context: a non-abstaining answer needs at least one
 * citation, and every citation must be a context ID. A violation fails validation inside generateObject, so the
 * repair loop tells the model which IDs are wrong, and a final violation is reported as invalid_output.
 */
export function answerSchemaFor(contextIds: readonly number[]) {
  const allowed = new Set(contextIds);
  const list = contextIds.join(', ');
  return Answer.superRefine((a, ctx) => {
    if (isAbstention(a)) {
      return;
    }
    if (a.citations.length === 0) {
      ctx.addIssue({
        code: 'custom',
        path: ['citations'],
        message: `found is true but citations is empty; cite the chunk IDs you used (allowed: ${list})`,
      });
      return;
    }
    const foreign = a.citations.filter((id) => !allowed.has(id));
    if (foreign.length > 0) {
      ctx.addIssue({
        code: 'custom',
        path: ['citations'],
        message: `citations ${foreign.join(', ')} are not chunk IDs from the context (allowed: ${list})`,
      });
    }
  });
}

/**
 * Longest accepted search query of the rewrite (M15) and expand (M16) stages; the golden questions are one sentence,
 * so more is a runaway output.
 */
export const MAX_QUERY_CHARS = 500;

const SearchQueryText = z.string().trim().min(1).max(MAX_QUERY_CHARS);

/**
 * Query rewrite (M15). Trimmed, non-empty and capped, so empty, whitespace-only and runaway outputs go through the
 * repair loop and end as invalid_output, which falls back to the original question.
 */
export const RewriteOutput = z.object({
  rewritten: SearchQueryText,
});
export type RewriteOutput = z.infer<typeof RewriteOutput>;

/**
 * Query expansion (M16). An empty list, blank or over-long strings and more than MAX_EXPANSIONS entries fail
 * validation (rejected, not truncated), go through the repair loop and end as invalid_output, which falls back to
 * the original question alone.
 */
export const ExpandOutput = z.object({
  queries: z.array(SearchQueryText).min(1).max(MAX_EXPANSIONS),
});
export type ExpandOutput = z.infer<typeof ExpandOutput>;

/**
 * Rerank output (M17): at least one entry, integer chunk IDs, scores within 0–MAX_RERANK_SCORE. Unknown, duplicate
 * and missing IDs are tolerated here and handled by applyRanking; a schema miss goes through the repair loop and ends
 * as invalid_output, which falls back to the search order.
 */
export const RerankOutput = z.object({
  ranking: z
    .array(
      z.object({
        chunk_id: z.number().int(),
        score: z.number().min(0).max(MAX_RERANK_SCORE),
      }),
    )
    .min(1),
});
export type RerankOutput = z.infer<typeof RerankOutput>;

/**
 * RerankOutput, refined against one request's candidates: a ranking that names none of them carries no judgement,
 * so it goes through the repair loop (with the allowed IDs in the message) instead of passing as a no-op.
 */
export function rerankSchemaFor(candidateIds: readonly number[]) {
  const allowed = new Set(candidateIds);
  const list = candidateIds.join(', ');
  return RerankOutput.superRefine((r, ctx) => {
    if (!r.ranking.some((e) => allowed.has(e.chunk_id))) {
      ctx.addIssue({
        code: 'custom',
        path: ['ranking'],
        message: `no chunk_id in ranking is a chunk ID from the context (allowed: ${list})`,
      });
    }
  });
}
