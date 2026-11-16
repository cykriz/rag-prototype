// The pipeline stages around retrieval (docs/PLAN.md M14–M17): one object travels from the CLIs and the eval config
// to buildAnswerDeps and the retrieval eval, so a new stage widens this type, not every call site. Agent limits: M18.
import type { SearchMode } from '../retrieval/search';

export const REWRITE_MODES = ['off', 'on'] as const;
export type RewriteMode = (typeof REWRITE_MODES)[number];
export const DEFAULT_REWRITE: RewriteMode = 'off';

export const EXPAND_MODES = ['off', 'on'] as const;
export type ExpandMode = (typeof EXPAND_MODES)[number];
export const DEFAULT_EXPAND: ExpandMode = 'off';
/** Most expansions per question (docs/PLAN.md M16, §7 "3 queries"); the question itself is searched as well. */
export const MAX_EXPANSIONS = 3;

export const RERANK_MODES = ['off', 'on'] as const;
export type RerankMode = (typeof RERANK_MODES)[number];
export const DEFAULT_RERANK: RerankMode = 'off';
/** Candidates retrieval hands to the reranker (docs/PLAN.md M17, §7 "top 20 to 5"); raised to k when k is larger. */
export const RERANK_CANDIDATES = 20;
/** Upper end of the rerank score (0–10); the schema and the prompt both use it. */
export const MAX_RERANK_SCORE = 10;
/**
 * Characters of chunk content in one rerank prompt, split evenly across the candidates (1200 each at 20). Sized in
 * M17: md-1024-128 chunks average about 2000 characters, 88 % of the dev evidence spans end within the first 1200
 * (47 % within 400); a 20-candidate prompt measured about 5.3k tokens, ample room in qwen2.5:7b's 32k context.
 * Headers come on top.
 */
export const RERANK_CONTEXT_CHARS = 24_000;

export interface PipelineConfig {
  search: SearchMode;
  rewrite: RewriteMode;
  expand: ExpandMode;
  rerank: RerankMode;
}

/** Rewrite and expand together are not evaluated (M16); their order would be a guess, so the combination is refused. */
export const STAGE_CONFLICT = 'rewrite and expand cannot both be on (the combination is not evaluated)';

/**
 * True for a stage combination the pipeline does not support; the one rule behind assertPipeline and RunConfig.
 * Rerank runs after retrieval whatever came before it, so it combines with either query stage.
 */
export const stagesConflict = (p: Pick<PipelineConfig, 'rewrite' | 'expand'>): boolean =>
  p.rewrite === 'on' && p.expand === 'on';

/** Throws on a stage combination the pipeline does not support. */
export function assertPipeline(p: Pick<PipelineConfig, 'rewrite' | 'expand'>): void {
  if (stagesConflict(p)) {
    throw new Error(STAGE_CONFLICT);
  }
}

/**
 * The chat-model stages in pipeline order: rewrite and expand before retrieval (M15, M16), rerank after it (M17).
 * The one list behind pipeline names, fallback counts, summaries and the chat-model pre-flight.
 */
export const CHAT_STAGES = ['rewrite', 'expand', 'rerank'] as const;
export type ChatStage = (typeof CHAT_STAGES)[number];

/** The stages that are on, in pipeline order. */
export const activeStages = (p: Pick<PipelineConfig, ChatStage>): ChatStage[] =>
  CHAT_STAGES.filter((stage) => p[stage] === 'on');

/** How many hits retrieval fetches: k, or max(k, RERANK_CANDIDATES) with rerank on (the reranker keeps k of them). */
export const retrievalDepth = (p: Pick<PipelineConfig, 'rerank'>, k: number): number =>
  p.rerank === 'on' ? Math.max(k, RERANK_CANDIDATES) : k;

/**
 * rag_requests.pipeline: `baseline` for plain vector search (as before M14), otherwise the search mode followed by
 * the active stages, e.g. `fts`, `vector+rewrite`, `hybrid+expand`, `vector+rerank`, `vector+expand+rerank`.
 */
export function pipelineName(p: PipelineConfig): string {
  const stages = activeStages(p);
  if (p.search === 'vector' && stages.length === 0) {
    return 'baseline';
  }
  return [p.search, ...stages].join('+');
}

/** Hard limit of executed search_docs calls per agent request (docs/PLAN.md M18a). */
export const MAX_TOOL_CALLS = 4;
/**
 * Largest k the agent may request per search_docs call; larger requests are clamped. 4 calls × 10 md-1024-128 chunks
 * (about 2000 characters each) stay near 20k tokens, inside qwen2.5:7b's 32k context.
 */
export const MAX_TOOL_K = 10;

/** rag_requests.pipeline of the agent (M18): `agent` with vector search as the tool, `<search>+agent` otherwise. */
export const agentPipelineName = (search: SearchMode): string =>
  search === 'vector' ? 'agent' : `${search}+agent`;
