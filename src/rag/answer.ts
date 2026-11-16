// Retrieve, rerank (M17), build context, generate, validate citations, log the request (docs/PLAN.md M9).
import { randomUUID } from 'node:crypto';
import type { Document } from '@langchain/core/documents';
import type { Sql } from 'postgres';
import type { Config } from '../config';
import { createChatProvider, createEmbeddingProvider } from '../llm/registry';
import { validate } from '../llm/structured';
import type { ChatProvider, ObjectResult } from '../llm/types';
import { PostgresLlmCallSink, PostgresRagRequestSink, type RagRequestSink } from '../obs/logger';
import { type ChunkMetadata, PgRetriever } from '../retrieval/pgRetriever';
import { type ExpandRecord, type Expander, expanderFor, expandRecord, searchQueries } from './expand';
import { type PipelineConfig, pipelineName, retrievalDepth } from './pipeline';
import { ANSWER_SEED, ANSWER_TEMPERATURE, buildAnswerMessages, formatContext, NOT_FOUND } from './prompts';
import {
  documentCandidate,
  type Reranker,
  type RerankRecord,
  type RerankTrace,
  rerankerFor,
  rerankItems,
  rerankRecord,
} from './rerank';
import { type RewriteRecord, type Rewriter, rewriteRecord, rewriterFor } from './rewrite';
import { Answer, answerSchemaFor, isAbstention } from './schemas';

export interface AnswerDeps {
  /**
   * PgRetriever fits; the depth and the search mode live in the retriever (k, or the rerank candidates). invokeMany
   * fuses several queries (M16).
   */
  retriever: {
    invoke(question: string): Promise<Document<ChunkMetadata>[]>;
    invokeMany(queries: readonly string[]): Promise<Document<ChunkMetadata>[]>;
  };
  chat: ChatProvider;
  /** Rewrites the question into the retrieval query (M15); absent = retrieve with the question itself. */
  rewrite?: Rewriter;
  /** Expands the question into extra search queries (M16); absent = one query. Never together with rewrite. */
  expand?: Expander;
  /** Reranks the retrieved candidates and keeps the top k (M17); absent = the retriever's hits as they are. */
  rerank?: { reranker: Reranker; k: number };
  /** Must equal the requestId of the llm_calls sinks behind retriever and chat; buildAnswerDeps guarantees it. */
  requestId: string;
  runId: string;
  pipeline: string;
  /** Golden dataset question ID for eval runs (rag_requests.question_id); absent for ad hoc questions. */
  questionId?: string;
  requestSink: RagRequestSink;
}

export interface Citation {
  chunkId: number;
  sourcePath: string;
  headingPath: string;
}

export interface AnswerResult {
  requestId: string;
  /** null when the model output was unusable even after repairs. */
  found: boolean | null;
  answer: string | null;
  citations: Citation[];
  /** Cited IDs outside the context, removed from the final answer. */
  droppedCitations: number[];
  /** The chunks in the context, in context order (reranked and cut to k with rerank on). */
  retrievedIds: number[];
  /** The query retrieval used and whether it fell back to the question; null without the rewrite stage. */
  rewrite: RewriteRecord | null;
  /** The expansions searched besides the question and whether the stage fell back; null without the expand stage. */
  expand: ExpandRecord | null;
  /** Candidates, rerank order, scores and whether the stage fell back; null without the rerank stage. */
  rerank: RerankRecord | null;
  /** The context text the model saw (formatContext of the retrieved chunks); '' when nothing was retrieved. */
  context: string;
  /** Status of the generate call: invalid_output when the last repair round still failed validation. */
  status: 'ok' | 'invalid_output';
  repairs: number;
  /** Whole request: rewrite or expand, retrieval, rerank, generation and validation. */
  latencyMs: number;
}

/** A failure during retrieval (embedding call or search); `cause` is the original error. */
export class RetrievalError extends Error {
  constructor(cause: unknown) {
    super(cause instanceof Error ? cause.message : String(cause), { cause });
    this.name = 'RetrievalError';
  }
}

/**
 * The Answer of a generateObject call: the validated value, or for invalid_output the base Answer salvaged from the
 * final raw output (null when even that does not parse), so citations outside the context can still be dropped.
 */
export function modelAnswer(res: ObjectResult<Answer>): Answer | null {
  if (res.status === 'ok') {
    return res.value;
  }
  const v = validate(res.raw, Answer);
  return v.ok ? v.value : null;
}

/** The abstention when nothing was retrieved: no model call (answer() and the agent, M18). */
export const NO_CONTEXT_ANSWER: Answer = { found: false, answer: NOT_FOUND, citations: [] };

export type AnswerFields = Pick<AnswerResult, 'found' | 'answer' | 'citations' | 'droppedCitations'>;

/**
 * Maps the model's Answer onto the result: null → unusable (found null), an abstention → NOT_FOUND, otherwise the
 * unique citations resolved against `byId` (the chunks the model saw); IDs outside it go to droppedCitations. Shared by
 * answer() and the agent (M18), whose `byId` is the union of its tool results.
 */
export function answerFields(model: Answer | null, byId: ReadonlyMap<number, ChunkMetadata>): AnswerFields {
  if (model === null) {
    return { found: null, answer: null, citations: [], droppedCitations: [] };
  }
  if (isAbstention(model)) {
    return { found: false, answer: NOT_FOUND, citations: [], droppedCitations: [] };
  }
  const citations: Citation[] = [];
  const droppedCitations: number[] = [];
  for (const id of new Set(model.citations)) {
    const meta = byId.get(id);
    if (meta) {
      citations.push({ chunkId: id, sourcePath: meta.source_path, headingPath: meta.heading_path });
    } else {
      droppedCitations.push(id);
    }
  }
  // found stays true even with zero valid citations: the citation-validity metric (M12) must see it.
  return { found: true, answer: model.answer, citations, droppedCitations };
}

export async function answer(question: string, deps: AnswerDeps): Promise<AnswerResult> {
  const start = performance.now();
  // The rewrite feeds retrieval only; the answer prompt keeps the user's question. Errors surface as ChatStageError.
  const rewrite = deps.rewrite ? await deps.rewrite(question) : null;
  // Expansion feeds retrieval only, like the rewrite; ChatStageError again, outside the RetrievalError wrap.
  const expansion = deps.expand ? await deps.expand(question) : null;
  let docs: Document<ChunkMetadata>[];
  try {
    docs = expansion
      ? await deps.retriever.invokeMany(searchQueries(question, expansion))
      : await deps.retriever.invoke(rewrite?.query ?? question);
  } catch (err) {
    throw new RetrievalError(err);
  }
  // After the RetrievalError wrap: a rerank provider error is a ChatStageError, like rewrite and expand. The reranker
  // judges the candidates against the user's question, not a rewrite or expansion.
  let reranked: RerankTrace | null = null;
  if (deps.rerank) {
    const { kept, trace } = await rerankItems(
      deps.rerank.reranker,
      question,
      docs,
      documentCandidate,
      deps.rerank.k,
    );
    docs = kept;
    reranked = trace;
  }
  const retrievedIds = docs.map((d) => d.metadata.chunk_id);
  const byId = new Map(docs.map((d) => [d.metadata.chunk_id, d.metadata]));
  const context = formatContext(docs);

  let model: Answer | null;
  let status: AnswerResult['status'] = 'ok';
  let repairs = 0;
  if (docs.length === 0) {
    // Nothing to ground an answer on; no model call.
    model = NO_CONTEXT_ANSWER;
  } else {
    const res: ObjectResult<Answer> = await deps.chat.generateObject(
      {
        messages: buildAnswerMessages(question, context),
        temperature: ANSWER_TEMPERATURE,
        seed: ANSWER_SEED,
      },
      answerSchemaFor(retrievedIds),
    );
    repairs = res.repairs;
    status = res.status;
    model = modelAnswer(res);
  }

  const result = answerFields(model, byId);
  const latencyMs = await writeRequest(deps, start, retrievedIds, result.found);
  return {
    requestId: deps.requestId,
    ...result,
    retrievedIds,
    rewrite: rewriteRecord(rewrite),
    expand: expandRecord(expansion),
    rerank: rerankRecord(reranked),
    context,
    status,
    repairs,
    latencyMs,
  };
}

/** What identifies a request in rag_requests; shared by AnswerDeps and the agent's deps (M18). */
export type RequestMeta = Pick<AnswerDeps, 'requestId' | 'runId' | 'pipeline' | 'questionId' | 'requestSink'>;

/** Writes the request's rag_requests row with the latency since `start` (performance.now()) and returns it. */
export async function writeRequest(
  meta: RequestMeta,
  start: number,
  retrievedIds: number[],
  found: boolean | null,
): Promise<number> {
  const latencyMs = Math.round(performance.now() - start);
  await meta.requestSink.write({
    requestId: meta.requestId,
    runId: meta.runId,
    questionId: meta.questionId ?? null,
    pipeline: meta.pipeline,
    totalLatencyMs: latencyMs,
    retrievedIds,
    found,
  });
  return latencyMs;
}

/**
 * One requestId with the llm_calls sink, both providers and the rag_requests sink behind it, so every row of a
 * request carries the same ID; shared by buildAnswerDeps and buildAgentDeps. Throws before any model call when a
 * model has no price (registry).
 */
export function requestWiring(sql: Sql, config: Config, runId: string) {
  const requestId = randomUUID();
  const sink = new PostgresLlmCallSink(sql, { runId, requestId });
  return {
    requestId,
    embedder: createEmbeddingProvider(config, sink),
    chat: createChatProvider(config, sink),
    requestSink: new PostgresRagRequestSink(sql),
  };
}

/**
 * Per-request wiring: one requestId shared by the rewrite/expand, embed, rerank and generate llm_calls rows and the
 * rag_requests row; the pipeline name comes from `pipeline`, which assertPipeline has accepted.
 * Throws before any model call when a model has no price (registry).
 */
export function buildAnswerDeps(
  sql: Sql,
  config: Config,
  opts: {
    chunkConfig: string;
    k: number;
    pipeline: PipelineConfig;
    runId: string;
    questionId?: string;
  },
): AnswerDeps {
  const { requestId, embedder, chat, requestSink } = requestWiring(sql, config, opts.runId);
  return {
    retriever: new PgRetriever({
      sql,
      embedder,
      chunkConfig: opts.chunkConfig,
      k: retrievalDepth(opts.pipeline, opts.k),
      search: opts.pipeline.search,
    }),
    chat,
    ...(opts.pipeline.rewrite === 'on' && { rewrite: rewriterFor(chat) }),
    ...(opts.pipeline.expand === 'on' && { expand: expanderFor(chat) }),
    ...(opts.pipeline.rerank === 'on' && { rerank: { reranker: rerankerFor(chat), k: opts.k } }),
    requestId,
    runId: opts.runId,
    pipeline: pipelineName(opts.pipeline),
    ...(opts.questionId !== undefined && { questionId: opts.questionId }),
    requestSink,
  };
}
