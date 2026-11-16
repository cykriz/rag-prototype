// LLM reranking (docs/PLAN.md M17): retrieval fetches max(k, 20) candidates, one structured call scores them over
// truncated chunks, the top k are kept. Unknown IDs are ignored, duplicates keep their first entry, missing
// candidates are appended in search order. Invalid output after the repair rounds falls back to the search order; a
// provider error surfaces as ChatStageError.
import type { Document } from '@langchain/core/documents';
import type { ChatProvider } from '../llm/types';
import type { ChunkMetadata } from '../retrieval/pgRetriever';
import { buildRerankMessages, RERANK_SEED, RERANK_TEMPERATURE, type RerankCandidate } from './prompts';
import { type RerankOutput, rerankSchemaFor } from './schemas';
import { runChatStage } from './stage';

export interface RerankTrace {
  /** Candidate chunk IDs in search order. */
  candidates: number[];
  /** Every candidate, reranked: model score descending (ties in model order), then the missing ones in search order. */
  order: number[];
  /** The model score per entry of `order`; null for an appended candidate (and for all on fallback or skip). */
  scores: (number | null)[];
  /** Ranked IDs that are no candidate, ignored. */
  unknown: number;
  /** Repeated IDs, ignored after their first entry. */
  duplicates: number;
  /** Candidates the ranking left out, appended. */
  missing: number;
  /** True when the output stayed invalid after the repairs: `order` is the search order. */
  fallback: boolean;
  repairs: number;
}

/** The per-question part of a trace that result files record. */
export type RerankRecord = Omit<RerankTrace, 'repairs'>;

/** The one injection point of the rerank stage, shared by the answer path, the retrieval eval and retrieve. */
export type Reranker = (question: string, candidates: readonly RerankCandidate[]) => Promise<RerankTrace>;

/** The search order unchanged, without scores: fallback, or nothing to rank. */
const unranked = (candidates: number[], fallback: boolean, repairs: number): RerankTrace => ({
  candidates,
  order: [...candidates],
  scores: candidates.map(() => null),
  unknown: 0,
  duplicates: 0,
  missing: 0,
  fallback,
  repairs,
});

/** Pure: the candidates reordered by a ranking (see RerankTrace.order), with the counts of what was ignored. */
export function applyRanking(
  candidates: readonly number[],
  ranking: RerankOutput['ranking'],
): Pick<RerankTrace, 'order' | 'scores' | 'unknown' | 'duplicates' | 'missing'> {
  const allowed = new Set(candidates);
  const seen = new Set<number>();
  const ranked: { id: number; score: number }[] = [];
  let unknown = 0;
  let duplicates = 0;
  for (const { chunk_id: id, score } of ranking) {
    if (!allowed.has(id)) {
      unknown++;
    } else if (seen.has(id)) {
      duplicates++;
    } else {
      seen.add(id);
      ranked.push({ id, score });
    }
  }
  // Array.prototype.sort is stable: equal scores keep the model's order.
  ranked.sort((a, b) => b.score - a.score);
  const rest = candidates.filter((id) => !seen.has(id));
  return {
    order: [...ranked.map((r) => r.id), ...rest],
    scores: [...ranked.map((r) => r.score), ...rest.map(() => null)],
    unknown,
    duplicates,
    missing: rest.length,
  };
}

/** Logged as operation `rerank` by LoggedChatProvider; fewer than two candidates make no call. */
export async function rerankChunks(
  chat: ChatProvider,
  question: string,
  candidates: readonly RerankCandidate[],
): Promise<RerankTrace> {
  const ids = candidates.map((c) => c.chunkId);
  if (ids.length < 2) {
    return unranked(ids, false, 0);
  }
  const res = await runChatStage(
    chat,
    'rerank',
    {
      messages: buildRerankMessages(question, candidates),
      temperature: RERANK_TEMPERATURE,
      seed: RERANK_SEED,
    },
    rerankSchemaFor(ids),
  );
  if (res.status !== 'ok') {
    return unranked(ids, true, res.repairs);
  }
  return { candidates: ids, ...applyRanking(ids, res.value.ranking), fallback: false, repairs: res.repairs };
}

export const rerankerFor =
  (chat: ChatProvider): Reranker =>
  (question, candidates) =>
    rerankChunks(chat, question, candidates);

export const rerankRecord = (t: RerankTrace | null): RerankRecord | null => {
  if (t === null) {
    return null;
  }
  const { repairs: _, ...record } = t;
  return record;
};

/** A retrieved Document as a rerank candidate (answer path, retrieve). */
export const documentCandidate = (d: Document<ChunkMetadata>): RerankCandidate => ({
  chunkId: d.metadata.chunk_id,
  sourcePath: d.metadata.source_path,
  headingPath: d.metadata.heading_path,
  content: d.pageContent,
});

/**
 * Reranks `items` (search hits, documents) and keeps the top k in rerank order: the one reorder-and-cut path of the
 * answer path, the retrieval eval and retrieve.
 */
export async function rerankItems<T>(
  reranker: Reranker,
  question: string,
  items: readonly T[],
  toCandidate: (item: T) => RerankCandidate,
  k: number,
): Promise<{ kept: T[]; trace: RerankTrace }> {
  const pairs = items.map((item) => ({ item, candidate: toCandidate(item) }));
  const trace = await reranker(
    question,
    pairs.map((p) => p.candidate),
  );
  const byId = new Map(pairs.map((p) => [p.candidate.chunkId, p.item]));
  const kept = trace.order.slice(0, k).map((id) => {
    const item = byId.get(id);
    if (item === undefined) {
      throw new Error(`rerank: chunk ${id} is no candidate`);
    }
    return item;
  });
  return { kept, trace };
}
