// Reciprocal rank fusion (docs/PLAN.md M14). hybrid.ts runs the same formula in SQL and binds RRF_K from here; the
// hybrid integration test checks that both agree.

/** Rank offset of RRF (Cormack et al.); damps the weight of the first ranks. */
export const RRF_K = 60;

export interface FusedHit {
  chunkId: number;
  score: number;
}

/**
 * Fuses ranked id lists (best first, ranks from 1): score = Σ 1/(RRF_K + rank) over the lists an id appears in.
 * Sorted by score descending, then chunk id ascending, as the SQL in hybrid.ts; cut to `limit`.
 */
export function rrf(lists: readonly (readonly number[])[], limit: number): FusedHit[] {
  const scores = new Map<number, number>();
  for (const list of lists) {
    list.forEach((id, i) => {
      scores.set(id, (scores.get(id) ?? 0) + 1 / (RRF_K + i + 1));
    });
  }
  return [...scores]
    .map(([chunkId, score]) => ({ chunkId, score }))
    .sort((a, b) => b.score - a.score || a.chunkId - b.chunkId)
    .slice(0, limit);
}
