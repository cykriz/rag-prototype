// Adapter 1 (ADR 0004): the openai package against an OpenAI-compatible endpoint, Ollama by default.
import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import type { z } from 'zod';
import type { EmbeddingModelConfig } from '../config';
import { type RetryOptions, withRetry } from './retry';
import { generateObject, type RawCompletion } from './structured';
import type {
  ChatProvider,
  EmbeddingProvider,
  EmbedResult,
  GenerateRequest,
  GenerateResult,
  Message,
  ObjectResult,
  ToolCall,
  Usage,
} from './types';

type ChatParams = OpenAI.Chat.ChatCompletionCreateParamsNonStreaming;

export interface ClientOptions {
  baseURL: string;
  /** Test hook: stub fetch for request mapping tests. */
  fetch?: typeof fetch;
  retry?: RetryOptions;
}

const TIMEOUT_MS = 120_000;

function createClient(opts: ClientOptions): OpenAI {
  // maxRetries: 0 so that only our retry.ts runs (no double retries).
  return new OpenAI({
    baseURL: opts.baseURL,
    apiKey: 'ollama',
    maxRetries: 0,
    timeout: TIMEOUT_MS,
    ...(opts.fetch && { fetch: opts.fetch }),
  });
}

function toOpenAIMessage(m: Message): OpenAI.Chat.ChatCompletionMessageParam {
  switch (m.role) {
    case 'tool':
      return { role: 'tool', tool_call_id: m.toolCallId, content: m.content };
    case 'assistant':
      return {
        role: 'assistant',
        content: m.content,
        ...(m.toolCalls?.length && {
          tool_calls: m.toolCalls.map((c) => ({
            id: c.id,
            type: 'function' as const,
            function: { name: c.name, arguments: c.arguments },
          })),
        }),
      };
    default:
      return m;
  }
}

function toParams(model: string, req: GenerateRequest, messages = req.messages): ChatParams {
  return {
    model,
    messages: messages.map(toOpenAIMessage),
    ...(req.temperature !== undefined && { temperature: req.temperature }),
    ...(req.seed !== undefined && { seed: req.seed }),
    ...(req.maxTokens !== undefined && { max_tokens: req.maxTokens }),
    ...(req.tools?.length && {
      tools: req.tools.map((t) => ({
        type: 'function' as const,
        function: { name: t.name, description: t.description, parameters: t.parameters },
      })),
    }),
  };
}

const chatUsage = (u: OpenAI.CompletionUsage | undefined): Usage | undefined =>
  u ? { inputTokens: u.prompt_tokens, outputTokens: u.completion_tokens } : undefined;

export class OpenAICompatibleChatProvider implements ChatProvider {
  private readonly client: OpenAI;

  constructor(private readonly opts: ClientOptions & { chatModel: string; judgeModel: string }) {
    this.client = createClient(opts);
  }

  private async complete(params: ChatParams) {
    const { value, attempts } = await withRetry(
      () => this.client.chat.completions.create(params),
      this.opts.retry,
    );
    const usage = chatUsage(value.usage);
    return { message: value.choices[0]?.message, model: value.model || params.model, attempts, usage };
  }

  async generate(req: GenerateRequest): Promise<GenerateResult> {
    const { message, model, attempts, usage } = await this.complete(toParams(this.opts.chatModel, req));
    const toolCalls: ToolCall[] = (message?.tool_calls ?? []).flatMap((c) =>
      c.type === 'function' ? [{ id: c.id, name: c.function.name, arguments: c.function.arguments }] : [],
    );
    return { text: message?.content ?? null, toolCalls, model, attempts, ...(usage && { usage }) };
  }

  private object<T>(model: string, req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>> {
    // Path (b) from docs/SPIKE.md: zodResponseFormat request; validation and repair in structured.ts.
    const responseFormat = zodResponseFormat(schema, 'output');
    const complete = async (messages: Message[]): Promise<RawCompletion> => {
      const res = await this.complete({ ...toParams(model, req, messages), response_format: responseFormat });
      return {
        content: res.message?.content ?? null,
        model: res.model,
        attempts: res.attempts,
        usage: res.usage,
      };
    };
    return generateObject(complete, req.messages, schema);
  }

  generateObject<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>> {
    return this.object(this.opts.chatModel, req, schema);
  }

  judge<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>> {
    return this.object(this.opts.judgeModel, req, schema);
  }
}

export class OpenAICompatibleEmbeddingProvider implements EmbeddingProvider {
  private readonly client: OpenAI;
  readonly dim: number;

  constructor(
    readonly model: string,
    private readonly config: EmbeddingModelConfig,
    private readonly opts: ClientOptions,
  ) {
    this.client = createClient(opts);
    this.dim = config.dim;
  }

  async embed(texts: string[], kind: 'document' | 'query'): Promise<EmbedResult> {
    // Task prefixes (e.g. nomic's search_document: / search_query:) come from config/models.json.
    const prefix = kind === 'document' ? this.config.documentPrefix : this.config.queryPrefix;
    const { value, attempts } = await withRetry(
      () =>
        this.client.embeddings.create({
          model: this.model,
          input: texts.map((t) => prefix + t),
          encoding_format: 'float', // docs/SPIKE.md M1: only float was verified against Ollama
        }),
      this.opts.retry,
    );
    const vectors = [...value.data].sort((a, b) => a.index - b.index).map((d) => d.embedding);
    if (vectors.length !== texts.length) {
      throw new Error(`${this.model}: expected ${texts.length} embeddings, got ${vectors.length}`);
    }
    for (const v of vectors) {
      if (v.length !== this.dim) {
        throw new Error(`${this.model}: expected dim ${this.dim}, got ${v.length}`);
      }
    }
    const usage = value.usage ? { inputTokens: value.usage.prompt_tokens, outputTokens: 0 } : undefined;
    return { vectors, model: this.model, attempts, ...(usage && { usage }) };
  }
}
