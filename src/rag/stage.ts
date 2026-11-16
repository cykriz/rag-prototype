// Chat-model stages around retrieval (docs/PLAN.md M15 rewrite, M16 expand, M17 rerank): one structured call per
// question, a provider error wrapped with the stage name, so callers handle every stage with one instanceof.
import type { z } from 'zod';
import type { ChatProvider, GenerateRequest, ObjectResult } from '../llm/types';
import type { ChatStage } from './pipeline';

/** A provider failure during a chat stage (chat model); `cause` is the original error. */
export class ChatStageError extends Error {
  constructor(
    readonly stage: ChatStage,
    cause: unknown,
  ) {
    super(cause instanceof Error ? cause.message : String(cause), { cause });
    this.name = 'ChatStageError';
  }
}

/**
 * generateObject tagged with the stage as operation (llm_calls); invalid output comes back as a result for the stage
 * to fall back on, a provider error is thrown as ChatStageError.
 */
export async function runChatStage<T>(
  chat: ChatProvider,
  stage: ChatStage,
  req: Omit<GenerateRequest, 'operation'>,
  schema: z.ZodType<T>,
): Promise<ObjectResult<T>> {
  try {
    return await chat.generateObject({ ...req, operation: stage }, schema);
  } catch (err) {
    throw new ChatStageError(stage, err);
  }
}
