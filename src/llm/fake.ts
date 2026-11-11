// Deterministic fake providers for tests and CI (docs/PLAN.md §9). No network, no model.
import { createHash } from 'node:crypto';
import type { z } from 'zod';
import { generateObject } from './structured';
import type {
  ChatProvider,
  EmbeddingProvider,
  EmbedResult,
  GenerateRequest,
  GenerateResult,
  Message,
  ObjectResult,
} from './types';

export const FAKE_CHAT_MODEL = 'fake-chat';
export const FAKE_JUDGE_MODEL = 'fake-judge';

/** One scripted reply: raw content, or an error to throw. */
export type ScriptedReply = string | Error;

/**
 * Replies come from one shared queue, consumed in call order by generate, generateObject and judge.
 * Without a script, generate echoes the last user message; generateObject and judge throw.
 */
export class FakeChatProvider implements ChatProvider {
  /** Every message list the fake received, for assertions. */
  readonly calls: Message[][] = [];

  constructor(private readonly script: ScriptedReply[] = []) {}

  private next(messages: Message[]): string | undefined {
    this.calls.push(messages);
    const reply = this.script.shift();
    if (reply instanceof Error) {
      throw reply;
    }
    return reply;
  }

  async generate(req: GenerateRequest): Promise<GenerateResult> {
    const scripted = this.next(req.messages);
    const lastUser = req.messages.findLast((m) => m.role === 'user')?.content ?? '';
    return { text: scripted ?? `fake: ${lastUser}`, toolCalls: [], model: FAKE_CHAT_MODEL, attempts: 1 };
  }

  private object<T>(model: string, req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>> {
    return generateObject(
      async (messages) => {
        const content = this.next(messages);
        if (content === undefined) {
          throw new Error('FakeChatProvider: no scripted response');
        }
        return { content, model, attempts: 1 };
      },
      req.messages,
      schema,
    );
  }

  generateObject<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>> {
    return this.object(FAKE_CHAT_MODEL, req, schema);
  }

  judge<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>> {
    return this.object(FAKE_JUDGE_MODEL, req, schema);
  }
}

/** sha256 of the text expanded to `dim` floats, L2-normalized. Identical text gives identical vectors. */
export function fakeEmbedding(text: string, dim: number): number[] {
  const values: number[] = [];
  for (let block = 0; values.length < dim; block++) {
    const digest = createHash('sha256').update(`${block}:${text}`).digest();
    for (const byte of digest) {
      if (values.length === dim) {
        break;
      }
      values.push(byte / 127.5 - 1);
    }
  }
  const norm = Math.hypot(...values) || 1;
  return values.map((v) => v / norm);
}

/** No task prefixes: a query equal to a chunk's text must get that chunk's vector (M8). */
export class FakeEmbeddingProvider implements EmbeddingProvider {
  constructor(
    readonly model: string,
    readonly dim: number,
  ) {}

  async embed(texts: string[], _kind: 'document' | 'query'): Promise<EmbedResult> {
    return { vectors: texts.map((t) => fakeEmbedding(t, this.dim)), model: this.model, attempts: 1 };
  }
}
