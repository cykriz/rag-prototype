// Provider interfaces (docs/PLAN.md M2, ADR 0001/0003). Every model call goes through these.
import type { z } from 'zod';

export interface ToolCall {
  id: string;
  name: string;
  /** Raw JSON string as returned by the model. */
  arguments: string;
}

export type Message =
  | { role: 'system'; content: string }
  | { role: 'user'; content: string }
  | { role: 'assistant'; content: string | null; toolCalls?: ToolCall[] }
  | { role: 'tool'; toolCallId: string; content: string };

export interface ToolDef {
  name: string;
  description: string;
  /** JSON schema of the arguments object. */
  parameters: Record<string, unknown>;
}

export interface GenerateRequest {
  messages: Message[];
  tools?: ToolDef[];
  temperature?: number;
  seed?: number;
  maxTokens?: number;
}

export interface Usage {
  inputTokens: number;
  outputTokens: number;
}

interface CallInfo {
  model: string;
  /** HTTP attempts including retries, summed over repair rounds. */
  attempts: number;
  /** Absent when the provider reports no usage (M3 estimates then). */
  usage?: Usage;
}

export interface GenerateResult extends CallInfo {
  text: string | null;
  toolCalls: ToolCall[];
}

export type ObjectResult<T> = CallInfo & { repairs: number } & (
    | { status: 'ok'; value: T }
    | { status: 'invalid_output'; error: string; raw: string | null }
  );

export interface EmbedResult extends CallInfo {
  vectors: number[][];
}

export interface ChatProvider {
  generate(req: GenerateRequest): Promise<GenerateResult>;
  generateObject<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>>;
  /** Same as generateObject, but on the judge model. */
  judge<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>>;
}

export interface EmbeddingProvider {
  readonly model: string;
  readonly dim: number;
  embed(texts: string[], kind: 'document' | 'query'): Promise<EmbedResult>;
}
