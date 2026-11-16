// LLM call logging (docs/PLAN.md M3): one record per provider call, written to a sink (Postgres sink in M4).
import type { Sql } from 'postgres';
import type { z } from 'zod';
import { costUsd, type Price } from '../llm/pricing';
import { attemptsOf } from '../llm/retry';
import {
  countMessageTokens,
  countOutputTokens,
  countTokens,
  resolveTokens,
  type TokenSource,
} from '../llm/tokens';
import type {
  ChatOperation,
  ChatProvider,
  EmbeddingProvider,
  EmbedResult,
  GenerateRequest,
  GenerateResult,
  ObjectResult,
  Usage,
} from '../llm/types';

/**
 * Mirrors the CHECK list of llm_calls.operation (docs/PLAN.md §4). The chat-model operations come from ChatOperation,
 * so a request can only carry an operation the table accepts.
 */
export type Operation = ChatOperation | 'embed' | 'judge' | 'agent_step';
export type CallStatus = 'ok' | 'error' | 'invalid_output';

/** One llm_calls row without id, run_id, request_id and created_at. */
export interface LlmCallRecord {
  operation: Operation;
  provider: string;
  model: string;
  inputTokens: number | null;
  outputTokens: number | null;
  tokenSource: TokenSource;
  latencyMs: number;
  costUsd: number;
  attempts: number;
  status: CallStatus;
}

export interface LlmCallSink {
  write(record: LlmCallRecord): Promise<void>;
}

/** What a record lacks to become an llm_calls row; fixed per sink. */
export interface LlmCallContext {
  runId: string;
  /** uuid of the RAG request (rag_requests.request_id); null for calls outside a request. */
  requestId?: string | null;
}

/** One llm_calls row in column names; id and created_at come from column defaults. */
export interface LlmCallRow {
  run_id: string;
  request_id: string | null;
  operation: Operation;
  provider: string;
  model: string;
  input_tokens: number | null;
  output_tokens: number | null;
  token_source: TokenSource;
  latency_ms: number;
  cost_usd: number;
  attempts: number;
  status: CallStatus;
}

export function toLlmCallRow(record: LlmCallRecord, ctx: LlmCallContext): LlmCallRow {
  return {
    run_id: ctx.runId,
    request_id: ctx.requestId ?? null,
    operation: record.operation,
    provider: record.provider,
    model: record.model,
    input_tokens: record.inputTokens,
    output_tokens: record.outputTokens,
    token_source: record.tokenSource,
    latency_ms: record.latencyMs,
    cost_usd: record.costUsd,
    attempts: record.attempts,
    status: record.status,
  };
}

/** Writes one llm_calls row per record. Constructed per run (and per request, once requests exist). */
export class PostgresLlmCallSink implements LlmCallSink {
  constructor(
    private readonly sql: Sql,
    private readonly ctx: LlmCallContext,
  ) {
    if (ctx.runId.trim() === '') {
      throw new Error('PostgresLlmCallSink needs a non-empty runId');
    }
  }

  async write(record: LlmCallRecord): Promise<void> {
    const row = toLlmCallRow(record, this.ctx);
    await this.sql`INSERT INTO llm_calls ${this.sql(row)}`;
  }
}

/** One rag_requests row (docs/PLAN.md §4/§8), written once per answered request. */
export interface RagRequestRecord {
  requestId: string;
  runId: string;
  /** Golden dataset question; null for ad hoc questions. */
  questionId: string | null;
  pipeline: string;
  totalLatencyMs: number;
  /** Chunk IDs in retrieval order. */
  retrievedIds: number[];
  /** null when the model output was unusable. */
  found: boolean | null;
}

export interface RagRequestSink {
  write(record: RagRequestRecord): Promise<void>;
}

/** Fail closed like the llm_calls sink: a write error fails the request. */
export class PostgresRagRequestSink implements RagRequestSink {
  constructor(private readonly sql: Sql) {}

  async write(r: RagRequestRecord): Promise<void> {
    await this.sql`
      INSERT INTO rag_requests (request_id, run_id, question_id, pipeline, total_latency_ms, retrieved_ids, found)
      VALUES (${r.requestId}, ${r.runId}, ${r.questionId}, ${r.pipeline}, ${r.totalLatencyMs},
              ${r.retrievedIds}::bigint[], ${r.found})`;
  }
}

interface Outcome<R> {
  model: string;
  attempts: number;
  usage?: Usage;
  status: Exclude<CallStatus, 'error'>;
  /** Output tokens; called only when the provider reported no usage. */
  estimateOutput: () => number;
  result: R;
}

interface CallTarget {
  sink: LlmCallSink;
  provider: string;
  operation: Operation;
  /** Model logged on the error path, where no result carries one. */
  model: string;
  price: Price;
  /**
   * Input tokens of the original request. For generateObject/judge with repairs this covers the first round
   * only, while attempts and provider usage span all rounds: a known underestimate.
   */
  estimateInput: () => number;
}

/** Times `call`, writes exactly one record, then returns the result or rethrows the provider error. */
async function logged<R>(target: CallTarget, call: () => Promise<Outcome<R>>): Promise<R> {
  const { sink, provider, operation, price } = target;
  const start = performance.now();
  const latency = () => Math.round(performance.now() - start);
  let outcome: Outcome<R>;
  try {
    outcome = await call();
  } catch (err) {
    // Read before estimating tokens, as on the success path, so latency excludes tokenizer time.
    const latencyMs = latency();
    const record: LlmCallRecord = {
      operation,
      provider,
      model: target.model,
      // The request is known, so its input is estimated; no response means no output and cost 0. Known
      // undercount: repair rounds that finished before a later round threw were billed but are not counted.
      inputTokens: target.estimateInput(),
      outputTokens: null,
      tokenSource: 'tiktoken_estimate',
      latencyMs,
      costUsd: 0,
      // Known undercount: a throw in a later repair round carries only that round's attempts, and a throw after
      // a successful withRetry (e.g. an embedding dim/count mismatch) carries none, so it falls back to 1.
      attempts: attemptsOf(err) ?? 1,
      status: 'error',
    };
    try {
      await sink.write(record);
    } catch (sinkErr) {
      throw new AggregateError([err, sinkErr], 'Provider call failed and the log sink failed too');
    }
    throw err;
  }
  const latencyMs = latency();
  const tokens = resolveTokens(outcome.usage, () => ({
    inputTokens: target.estimateInput(),
    outputTokens: outcome.estimateOutput(),
  }));
  // Fail closed: a lost record would understate cost and skew the KPIs, so a sink error fails the call.
  await sink.write({
    operation,
    provider,
    model: outcome.model,
    inputTokens: tokens.inputTokens,
    outputTokens: tokens.outputTokens,
    tokenSource: tokens.tokenSource,
    latencyMs,
    costUsd: costUsd(price, tokens),
    attempts: outcome.attempts,
    status: outcome.status,
  });
  return outcome.result;
}

export interface LoggedChatOptions {
  sink: LlmCallSink;
  provider: string;
  chatModel: string;
  judgeModel: string;
  chatPrice: Price;
  judgePrice: Price;
}

export class LoggedChatProvider implements ChatProvider {
  constructor(
    private readonly inner: ChatProvider,
    private readonly opts: LoggedChatOptions,
  ) {}

  private target(operation: Operation, model: string, price: Price, req: GenerateRequest): CallTarget {
    const { sink, provider } = this.opts;
    return { sink, provider, operation, model, price, estimateInput: () => countMessageTokens(req.messages) };
  }

  generate(req: GenerateRequest): Promise<GenerateResult> {
    return logged(
      this.target(req.operation ?? 'generate', this.opts.chatModel, this.opts.chatPrice, req),
      async () => {
        const res = await this.inner.generate(req);
        return {
          ...res,
          status: 'ok',
          estimateOutput: () => countOutputTokens(res.text, res.toolCalls),
          result: res,
        };
      },
    );
  }

  private object<T>(target: CallTarget, run: () => Promise<ObjectResult<T>>): Promise<ObjectResult<T>> {
    return logged(target, async () => {
      const res = await run();
      const output = res.status === 'ok' ? JSON.stringify(res.value) : res.raw;
      return {
        ...res,
        estimateOutput: () => countTokens(output ?? ''),
        result: res,
      };
    });
  }

  generateObject<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>> {
    const target = this.target(req.operation ?? 'generate', this.opts.chatModel, this.opts.chatPrice, req);
    return this.object(target, () => this.inner.generateObject(req, schema));
  }

  judge<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>> {
    const target = this.target('judge', this.opts.judgeModel, this.opts.judgePrice, req);
    return this.object(target, () => this.inner.judge(req, schema));
  }
}

export interface LoggedEmbeddingOptions {
  sink: LlmCallSink;
  provider: string;
  price: Price;
}

export class LoggedEmbeddingProvider implements EmbeddingProvider {
  readonly model: string;
  readonly dim: number;

  constructor(
    private readonly inner: EmbeddingProvider,
    private readonly opts: LoggedEmbeddingOptions,
  ) {
    this.model = inner.model;
    this.dim = inner.dim;
  }

  embed(texts: string[], kind: 'document' | 'query'): Promise<EmbedResult> {
    const { sink, provider, price } = this.opts;
    // Raw texts only: the task prefix the adapter adds is a few tokens and unknown here.
    const estimateInput = () => texts.reduce((n, t) => n + countTokens(t), 0);
    return logged(
      { sink, provider, operation: 'embed', model: this.model, price, estimateInput },
      async () => {
        const res = await this.inner.embed(texts, kind);
        return { ...res, status: 'ok', estimateOutput: () => 0, result: res };
      },
    );
  }
}
