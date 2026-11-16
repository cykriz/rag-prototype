// The pipeline stages around retrieval (docs/PLAN.md M14/M15): one object travels from the CLIs and the eval config
// to buildAnswerDeps and the retrieval eval, so a later stage (expand, rerank) widens this type, not every call site.
import type { SearchMode } from '../retrieval/search';

export const REWRITE_MODES = ['off', 'on'] as const;
export type RewriteMode = (typeof REWRITE_MODES)[number];
export const DEFAULT_REWRITE: RewriteMode = 'off';

export interface PipelineConfig {
  search: SearchMode;
  rewrite: RewriteMode;
}

/**
 * rag_requests.pipeline: `baseline` for plain vector search (as before M14), otherwise the search mode followed by
 * the active stages, e.g. `fts`, `vector+rewrite`, `hybrid+rewrite`.
 */
export function pipelineName(p: PipelineConfig): string {
  const stages = p.rewrite === 'on' ? ['rewrite'] : [];
  if (p.search === 'vector' && stages.length === 0) {
    return 'baseline';
  }
  return [p.search, ...stages].join('+');
}
