// Query rewriting (docs/PLAN.md M15): one structured call turns the question into a search query. Invalid output
// after the repair rounds falls back to the original question; a provider error surfaces as ChatStageError.
import type { ChatProvider } from '../llm/types';
import { buildRewriteMessages, REWRITE_SEED, REWRITE_TEMPERATURE } from './prompts';
import { RewriteOutput } from './schemas';
import { runChatStage } from './stage';

export interface RewriteTrace {
  /** What retrieval searches with: the rewrite, or the original question on fallback. */
  query: string;
  /** True when the output stayed invalid after the repairs. */
  fallback: boolean;
  repairs: number;
}

/** The per-question part of a trace that result files record. */
export type RewriteRecord = Pick<RewriteTrace, 'query' | 'fallback'>;

/** The one injection point of the rewrite stage, shared by the answer path and the retrieval eval. */
export type Rewriter = (question: string) => Promise<RewriteTrace>;

/** Logged as operation `rewrite` by LoggedChatProvider. */
export async function rewriteQuery(chat: ChatProvider, question: string): Promise<RewriteTrace> {
  const res = await runChatStage(
    chat,
    'rewrite',
    { messages: buildRewriteMessages(question), temperature: REWRITE_TEMPERATURE, seed: REWRITE_SEED },
    RewriteOutput,
  );
  return res.status === 'ok'
    ? { query: res.value.rewritten, fallback: false, repairs: res.repairs }
    : { query: question, fallback: true, repairs: res.repairs };
}

export const rewriterFor =
  (chat: ChatProvider): Rewriter =>
  (question) =>
    rewriteQuery(chat, question);

export const rewriteRecord = (t: RewriteTrace | null): RewriteRecord | null =>
  t === null ? null : { query: t.query, fallback: t.fallback };
