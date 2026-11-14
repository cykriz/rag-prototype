// Zod schemas for structured model output (docs/PLAN.md M9; Rewrite, Expansion, Rerank follow in phase 2).
import { z } from 'zod';
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
