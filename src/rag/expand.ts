// Query expansion (docs/PLAN.md M16): one structured call proposes up to MAX_EXPANSIONS extra search queries;
// retrieval searches the question and each expansion and fuses the lists with RRF. Invalid output after the repair
// rounds falls back to the question alone; a provider error surfaces as ChatStageError.
import type { ChatProvider } from '../llm/types';
import { buildExpandMessages, EXPAND_SEED, EXPAND_TEMPERATURE } from './prompts';
import { ExpandOutput } from './schemas';
import { runChatStage } from './stage';

export interface ExpandTrace {
  /** The expansions retrieval searches besides the question: deduped, in model order; [] on fallback. */
  queries: string[];
  /** Expansions removed as duplicates of the question or of an earlier expansion. */
  dropped: number;
  /** True when the output stayed invalid after the repairs. */
  fallback: boolean;
  repairs: number;
}

/** The per-question part of a trace that result files record. */
export type ExpandRecord = Pick<ExpandTrace, 'queries' | 'dropped' | 'fallback'>;

/** The one injection point of the expand stage, shared by the answer path and the retrieval eval. */
export type Expander = (question: string) => Promise<ExpandTrace>;

/** Comparison key for duplicates: whitespace collapsed, case folded. */
const dedupeKey = (q: string): string => q.trim().replace(/\s+/g, ' ').toLowerCase();

/** Drops expansions equal to the question or to an earlier expansion (dedupeKey); keeps the first spelling. */
export function dedupeExpansions(question: string, expansions: readonly string[]): string[] {
  const seen = new Set([dedupeKey(question)]);
  const kept: string[] = [];
  for (const q of expansions) {
    const key = dedupeKey(q);
    if (!seen.has(key)) {
      seen.add(key);
      kept.push(q);
    }
  }
  return kept;
}

/** Logged as operation `expand` by LoggedChatProvider. */
export async function expandQuery(chat: ChatProvider, question: string): Promise<ExpandTrace> {
  const res = await runChatStage(
    chat,
    'expand',
    { messages: buildExpandMessages(question), temperature: EXPAND_TEMPERATURE, seed: EXPAND_SEED },
    ExpandOutput,
  );
  if (res.status !== 'ok') {
    return { queries: [], dropped: 0, fallback: true, repairs: res.repairs };
  }
  const queries = dedupeExpansions(question, res.value.queries);
  return {
    queries,
    dropped: res.value.queries.length - queries.length,
    fallback: false,
    repairs: res.repairs,
  };
}

export const expanderFor =
  (chat: ChatProvider): Expander =>
  (question) =>
    expandQuery(chat, question);

export const expandRecord = (t: ExpandTrace | null): ExpandRecord | null =>
  t === null ? null : { queries: t.queries, dropped: t.dropped, fallback: t.fallback };

/** What retrieval searches: the question first, then the expansions; the question alone without a trace. */
export const searchQueries = (question: string, t: Pick<ExpandTrace, 'queries'> | null): string[] => [
  question,
  ...(t?.queries ?? []),
];
