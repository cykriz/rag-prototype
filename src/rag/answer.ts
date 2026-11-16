// Retrieve, build context, generate, validate citations, log the request (docs/PLAN.md M9).
import { randomUUID } from 'node:crypto';
import type { Document } from '@langchain/core/documents';
import type { Sql } from 'postgres';
import type { Config } from '../config';
import { createChatProvider, createEmbeddingProvider } from '../llm/registry';
import { validate } from '../llm/structured';
import type { ChatProvider, ObjectResult } from '../llm/types';
import { PostgresLlmCallSink, PostgresRagRequestSink, type RagRequestSink } from '../obs/logger';
import { type ChunkMetadata, PgRetriever } from '../retrieval/pgRetriever';
import { type PipelineConfig, pipelineName } from './pipeline';
import { ANSWER_SEED, ANSWER_TEMPERATURE, buildAnswerMessages, formatContext, NOT_FOUND } from './prompts';
import { type RewriteRecord, type Rewriter, rewriteRecord, rewriterFor } from './rewrite';
import { Answer, answerSchemaFor, isAbstention } from './schemas';

export interface AnswerDeps {
  /** PgRetriever fits; k and the search mode live in the retriever. */
  retriever: { invoke(question: string): Promise<Document<ChunkMetadata>[]> };
  chat: ChatProvider;
  /** Rewrites the question into the retrieval query (M15); absent = retrieve with the question itself. */
  rewrite?: Rewriter;
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
  retrievedIds: number[];
  /** The query retrieval used and whether it fell back to the question; null without the rewrite stage. */
  rewrite: RewriteRecord | null;
  /** The context text the model saw (formatContext of the retrieved chunks); '' when nothing was retrieved. */
  context: string;
  /** Status of the generate call: invalid_output when the last repair round still failed validation. */
  status: 'ok' | 'invalid_output';
  repairs: number;
  /** Whole request: rewrite, retrieval, generation and validation. */
  latencyMs: number;
}

/** A failure during retrieval (embedding call or search); `cause` is the original error. */
export class RetrievalError extends Error {
  constructor(cause: unknown) {
    super(cause instanceof Error ? cause.message : String(cause), { cause });
    this.name = 'RetrievalError';
  }
}

/** The base Answer from the final raw output, or null when even that does not parse. */
function salvage(raw: string | null): Answer | null {
  const v = validate(raw, Answer);
  return v.ok ? v.value : null;
}

export async function answer(question: string, deps: AnswerDeps): Promise<AnswerResult> {
  const start = performance.now();
  // The rewrite feeds retrieval only; the answer prompt keeps the user's question. Errors surface as RewriteError.
  const rewrite = deps.rewrite ? await deps.rewrite(question) : null;
  let docs: Document<ChunkMetadata>[];
  try {
    docs = await deps.retriever.invoke(rewrite?.query ?? question);
  } catch (err) {
    throw new RetrievalError(err);
  }
  const retrievedIds = docs.map((d) => d.metadata.chunk_id);
  const byId = new Map(docs.map((d) => [d.metadata.chunk_id, d.metadata]));
  const context = formatContext(docs);

  let model: Answer | null;
  let status: AnswerResult['status'] = 'ok';
  let repairs = 0;
  if (docs.length === 0) {
    // Nothing to ground an answer on; no model call.
    model = { found: false, answer: NOT_FOUND, citations: [] };
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
    model = res.status === 'ok' ? res.value : salvage(res.raw);
  }

  let result: Pick<AnswerResult, 'found' | 'answer' | 'citations' | 'droppedCitations'>;
  if (model === null) {
    result = { found: null, answer: null, citations: [], droppedCitations: [] };
  } else if (isAbstention(model)) {
    result = { found: false, answer: NOT_FOUND, citations: [], droppedCitations: [] };
  } else {
    const unique = [...new Set(model.citations)];
    const citations: Citation[] = [];
    const droppedCitations: number[] = [];
    for (const id of unique) {
      const meta = byId.get(id);
      if (meta) {
        citations.push({ chunkId: id, sourcePath: meta.source_path, headingPath: meta.heading_path });
      } else {
        droppedCitations.push(id);
      }
    }
    // found stays true even with zero valid citations: the citation-validity metric (M12) must see it.
    result = { found: true, answer: model.answer, citations, droppedCitations };
  }

  const latencyMs = Math.round(performance.now() - start);
  await deps.requestSink.write({
    requestId: deps.requestId,
    runId: deps.runId,
    questionId: deps.questionId ?? null,
    pipeline: deps.pipeline,
    totalLatencyMs: latencyMs,
    retrievedIds,
    found: result.found,
  });
  return {
    requestId: deps.requestId,
    ...result,
    retrievedIds,
    rewrite: rewriteRecord(rewrite),
    context,
    status,
    repairs,
    latencyMs,
  };
}

/**
 * Per-request wiring: one requestId shared by the rewrite, embed and generate llm_calls rows and the rag_requests row;
 * the pipeline name comes from `pipeline`.
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
  const requestId = randomUUID();
  const sink = new PostgresLlmCallSink(sql, { runId: opts.runId, requestId });
  const embedder = createEmbeddingProvider(config, sink);
  const chat = createChatProvider(config, sink);
  return {
    retriever: new PgRetriever({
      sql,
      embedder,
      chunkConfig: opts.chunkConfig,
      k: opts.k,
      search: opts.pipeline.search,
    }),
    chat,
    ...(opts.pipeline.rewrite === 'on' && { rewrite: rewriterFor(chat) }),
    requestId,
    runId: opts.runId,
    pipeline: pipelineName(opts.pipeline),
    ...(opts.questionId !== undefined && { questionId: opts.questionId }),
    requestSink: new PostgresRagRequestSink(sql),
  };
}
