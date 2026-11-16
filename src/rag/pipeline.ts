// The pipeline stages around retrieval (docs/PLAN.md M14–M16): one object travels from the CLIs and the eval config
// to buildAnswerDeps and the retrieval eval, so a later stage (rerank) widens this type, not every call site.
import type { SearchMode } from '../retrieval/search';

export const REWRITE_MODES = ['off', 'on'] as const;
export type RewriteMode = (typeof REWRITE_MODES)[number];
export const DEFAULT_REWRITE: RewriteMode = 'off';

export const EXPAND_MODES = ['off', 'on'] as const;
export type ExpandMode = (typeof EXPAND_MODES)[number];
export const DEFAULT_EXPAND: ExpandMode = 'off';
/** Most expansions per question (docs/PLAN.md M16, §7 "3 queries"); the question itself is searched as well. */
export const MAX_EXPANSIONS = 3;

export interface PipelineConfig {
  search: SearchMode;
  rewrite: RewriteMode;
  expand: ExpandMode;
}

/** Rewrite and expand together are not evaluated (M16); their order would be a guess, so the combination is refused. */
export const STAGE_CONFLICT = 'rewrite and expand cannot both be on (the combination is not evaluated)';

/** True for a stage combination the pipeline does not support; the one rule behind assertPipeline and RunConfig. */
export const stagesConflict = (p: Pick<PipelineConfig, 'rewrite' | 'expand'>): boolean =>
  p.rewrite === 'on' && p.expand === 'on';

/** Throws on a stage combination the pipeline does not support. */
export function assertPipeline(p: Pick<PipelineConfig, 'rewrite' | 'expand'>): void {
  if (stagesConflict(p)) {
    throw new Error(STAGE_CONFLICT);
  }
}

/**
 * The chat-model stages before retrieval (M15, M16) in pipeline order; the one list behind pipeline names, fallback
 * counts, summaries and the chat-model pre-flight.
 */
export const QUERY_STAGES = ['rewrite', 'expand'] as const;
export type QueryStage = (typeof QUERY_STAGES)[number];

/** The stages that are on, in pipeline order. */
export const activeStages = (p: Pick<PipelineConfig, QueryStage>): QueryStage[] =>
  QUERY_STAGES.filter((stage) => p[stage] === 'on');

/**
 * rag_requests.pipeline: `baseline` for plain vector search (as before M14), otherwise the search mode followed by
 * the active stages, e.g. `fts`, `vector+rewrite`, `hybrid+expand`.
 */
export function pipelineName(p: PipelineConfig): string {
  const stages = activeStages(p);
  if (p.search === 'vector' && stages.length === 0) {
    return 'baseline';
  }
  return [p.search, ...stages].join('+');
}
