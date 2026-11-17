// LangGraph agent (docs/PLAN.md M18a): the chat model calls search_docs up to MAX_TOOL_CALLS times, then answers
// through the Answer schema, its citations checked against the union of chunks its searches returned. Model calls go
// through our ChatProvider; messages are our own types, not LangChain chat models.
import type { Document } from '@langchain/core/documents';
import { Annotation, END, START, StateGraph } from '@langchain/langgraph';
import type { Sql } from 'postgres';
import type { Config } from '../config';
import { validate } from '../llm/structured';
import type { ChatProvider, Message, ToolCall } from '../llm/types';
import type { RagRequestSink } from '../obs/logger';
import {
  type AnswerResult,
  answerFields,
  modelAnswer,
  NO_CONTEXT_ANSWER,
  RetrievalError,
  requestWiring,
  writeRequest,
} from '../rag/answer';
import { agentPipelineName, MAX_TOOL_CALLS } from '../rag/pipeline';
import {
  AGENT_SEED,
  AGENT_TEMPERATURE,
  ANSWER_SEED,
  ANSWER_TEMPERATURE,
  buildAgentMessages,
  buildAnswerMessages,
  formatContext,
} from '../rag/prompts';
import { type Answer, answerSchemaFor } from '../rag/schemas';
import { type ChunkMetadata, PgRetriever } from '../retrieval/pgRetriever';
import type { SearchMode } from '../retrieval/search';
import { parseSearchArgs, searchDocsTool, searchResultContent, toolErrorContent } from './tools';

export interface AgentDeps {
  /** One search at depth k (PgRetriever.invoke); buildAgentDeps builds a retriever per call. */
  search(query: string, k: number): Promise<Document<ChunkMetadata>[]>;
  chat: ChatProvider;
  /** k of a search_docs call without k (the request's --k). */
  defaultK: number;
  /** Must equal the requestId of the llm_calls sinks behind search and chat; buildAgentDeps guarantees it. */
  requestId: string;
  runId: string;
  pipeline: string;
  questionId?: string;
  requestSink: RagRequestSink;
}

/** One executed search_docs call, for the trace (M18b) and the result file (M19). */
export interface AgentStep {
  /** null when the arguments did not parse. */
  query: string | null;
  /** k as the model sent it; null when absent or unparsable. */
  kRequested: number | null;
  /** k used after defaulting and clamping; null on an error. */
  k: number | null;
  /** Returned chunk IDs in search order; empty on an error. */
  chunkIds: number[];
  /** Why the call was not executed (unknown tool, bad arguments); null otherwise. */
  error: string | null;
}

export interface AgentResult extends AnswerResult {
  trace: AgentStep[];
  /** Executed search_docs calls (≤ MAX_TOOL_CALLS), errors included. */
  toolCalls: number;
  /** Tool calls the model asked for beyond the limit; not executed. */
  skippedToolCalls: number;
  /** The limit was reached, so the answer was forced without asking the agent again. */
  stoppedAtLimit: boolean;
  /** The agent's own final reply was no valid answer (or the limit was reached), so a structured answer call ran. */
  answerFallback: boolean;
}

const concat = <T>(a: T[], b: T[]): T[] => a.concat(b);
const sum = (a: number, b: number): number => a + b;

/** Deduplicated by chunk ID, first occurrence wins, so the union keeps the order the searches found the chunks in. */
function unionDocs(a: Document<ChunkMetadata>[], b: Document<ChunkMetadata>[]): Document<ChunkMetadata>[] {
  const seen = new Set(a.map((d) => d.metadata.chunk_id));
  return a.concat(b.filter((d) => !seen.has(d.metadata.chunk_id) && seen.add(d.metadata.chunk_id)));
}

const AgentState = Annotation.Root({
  messages: Annotation<Message[]>({ reducer: concat, default: () => [] }),
  /** Tool calls of the last agent turn, still to execute. */
  pending: Annotation<ToolCall[]>({ reducer: (_, b) => b, default: () => [] }),
  /** Text of the last agent turn without tool calls; null otherwise. */
  finalText: Annotation<string | null>({ reducer: (_, b) => b, default: () => null }),
  toolCalls: Annotation<number>({ reducer: sum, default: () => 0 }),
  skippedToolCalls: Annotation<number>({ reducer: sum, default: () => 0 }),
  retrieved: Annotation<Document<ChunkMetadata>[]>({ reducer: unionDocs, default: () => [] }),
  trace: Annotation<AgentStep[]>({ reducer: concat, default: () => [] }),
  final: Annotation<FinalAnswer | null>({ reducer: (_, b) => b, default: () => null }),
});
type State = typeof AgentState.State;

interface FinalAnswer {
  model: Answer | null;
  status: AnswerResult['status'];
  repairs: number;
  fallback: boolean;
}

/** Strips one surrounding Markdown code fence (```json … ```), which local models often add around JSON. */
export function stripFence(text: string): string {
  const m = /^\s*```[a-zA-Z]*\s*\n?([\s\S]*?)\n?\s*```\s*$/.exec(text);
  return m?.[1] ?? text;
}

/** agent → (tool calls) tools → agent …; no tool calls or the limit reached → answer → END. */
function buildGraph(question: string, deps: AgentDeps) {
  const tool = searchDocsTool(deps.defaultK);

  const agent = async (state: State): Promise<Partial<State>> => {
    const res = await deps.chat.generate({
      messages: state.messages,
      tools: [tool],
      operation: 'agent_step',
      temperature: AGENT_TEMPERATURE,
      seed: AGENT_SEED,
    });
    const reply: Message =
      res.toolCalls.length > 0
        ? { role: 'assistant', content: res.text, toolCalls: res.toolCalls }
        : { role: 'assistant', content: res.text ?? '' };
    return {
      messages: [reply],
      pending: res.toolCalls,
      finalText: res.toolCalls.length > 0 ? null : res.text,
    };
  };

  const tools = async (state: State): Promise<Partial<State>> => {
    const run = state.pending.slice(0, MAX_TOOL_CALLS - state.toolCalls);
    const messages: Message[] = [];
    const trace: AgentStep[] = [];
    let retrieved: Document<ChunkMetadata>[] = [];
    for (const call of run) {
      const args = parseSearchArgs(call, deps.defaultK);
      if (!args.ok) {
        trace.push({ query: null, kRequested: null, k: null, chunkIds: [], error: args.error });
        messages.push({ role: 'tool', toolCallId: call.id, content: toolErrorContent(args.error) });
        continue;
      }
      let docs: Document<ChunkMetadata>[];
      try {
        docs = await deps.search(args.query, args.k);
      } catch (err) {
        throw new RetrievalError(err);
      }
      trace.push({
        query: args.query,
        kRequested: args.kRequested,
        k: args.k,
        chunkIds: docs.map((d) => d.metadata.chunk_id),
        error: null,
      });
      messages.push({ role: 'tool', toolCallId: call.id, content: searchResultContent(docs) });
      retrieved = unionDocs(retrieved, docs);
    }
    return {
      messages,
      trace,
      retrieved,
      pending: [],
      toolCalls: run.length,
      skippedToolCalls: state.pending.length - run.length,
    };
  };

  const answer = async (state: State): Promise<Partial<State>> => {
    const ids = state.retrieved.map((d) => d.metadata.chunk_id);
    const schema = answerSchemaFor(ids);
    if (state.finalText !== null) {
      const own = validate(stripFence(state.finalText), schema);
      if (own.ok) {
        return { final: { model: own.value, status: 'ok', repairs: 0, fallback: false } };
      }
    }
    if (ids.length === 0) {
      // Nothing retrieved to ground an answer on; no model call (as in answer()).
      return {
        final: {
          model: NO_CONTEXT_ANSWER,
          status: 'ok',
          repairs: 0,
          fallback: false,
        },
      };
    }
    // Always the concise answer prompt: the agent is not evaluated with the complete style (agentConflict, M19c).
    const res = await deps.chat.generateObject(
      {
        messages: buildAnswerMessages(question, formatContext(state.retrieved)),
        operation: 'agent_step',
        temperature: ANSWER_TEMPERATURE,
        seed: ANSWER_SEED,
      },
      schema,
    );
    return { final: { model: modelAnswer(res), status: res.status, repairs: res.repairs, fallback: true } };
  };

  return new StateGraph(AgentState)
    .addNode('agent', agent)
    .addNode('tools', tools)
    .addNode('answer', answer)
    .addEdge(START, 'agent')
    .addConditionalEdges('agent', (s: State) => (s.pending.length > 0 ? 'tools' : 'answer'), [
      'tools',
      'answer',
    ])
    .addConditionalEdges('tools', (s: State) => (s.toolCalls >= MAX_TOOL_CALLS ? 'answer' : 'agent'), [
      'agent',
      'answer',
    ])
    .addEdge('answer', END)
    .compile();
}

/** Every node run is one superstep: MAX_TOOL_CALLS × (agent + tools) + answer, plus slack. A backstop only. */
const RECURSION_LIMIT = 2 * MAX_TOOL_CALLS + 3;

/**
 * Runs the agent for one question and writes its rag_requests row. Tool retrieval errors throw RetrievalError; chat
 * provider errors propagate unwrapped, as generate does in answer().
 */
export async function runAgent(question: string, deps: AgentDeps): Promise<AgentResult> {
  const start = performance.now();
  const graph = buildGraph(question, deps);
  const state = await graph.invoke(
    { messages: buildAgentMessages(question) },
    { recursionLimit: RECURSION_LIMIT },
  );
  const final = state.final;
  if (final === null) {
    throw new Error('agent graph ended without an answer');
  }
  const byId = new Map(state.retrieved.map((d) => [d.metadata.chunk_id, d.metadata]));
  const fields = answerFields(final.model, byId);
  const retrievedIds = [...byId.keys()];
  const latencyMs = await writeRequest(deps, start, retrievedIds, fields.found);
  return {
    requestId: deps.requestId,
    ...fields,
    retrievedIds,
    rewrite: null,
    expand: null,
    rerank: null,
    context: formatContext(state.retrieved),
    status: final.status,
    repairs: final.repairs,
    latencyMs,
    trace: state.trace,
    toolCalls: state.toolCalls,
    skippedToolCalls: state.skippedToolCalls,
    stoppedAtLimit: state.toolCalls >= MAX_TOOL_CALLS,
    answerFallback: final.fallback,
  };
}

/**
 * Per-request wiring like buildAnswerDeps: one requestId shared by the embed and agent_step llm_calls rows and the
 * rag_requests row. Throws before any model call when a model has no price (registry).
 */
export function buildAgentDeps(
  sql: Sql,
  config: Config,
  opts: { chunkConfig: string; k: number; search: SearchMode; runId: string; questionId?: string },
): AgentDeps {
  const { requestId, embedder, chat, requestSink } = requestWiring(sql, config, opts.runId);
  return {
    search: (query, k) =>
      new PgRetriever({ sql, embedder, chunkConfig: opts.chunkConfig, k, search: opts.search }).invoke(query),
    chat,
    defaultK: opts.k,
    requestId,
    runId: opts.runId,
    pipeline: agentPipelineName(opts.search),
    ...(opts.questionId !== undefined && { questionId: opts.questionId }),
    requestSink,
  };
}
