// Answer prompt: NOT_FOUND rule and citation format (docs/PLAN.md M9).
import type { Document } from '@langchain/core/documents';
import type { Message } from '../llm/types';
import { type ChunkMetadata, sourceLabel } from '../retrieval/pgRetriever';

export const NOT_FOUND = 'NOT_FOUND';
/** Fixed for reproducible runs (docs/PLAN.md §7). */
export const ANSWER_TEMPERATURE = 0;
export const ANSWER_SEED = 42;

export const ANSWER_SYSTEM_PROMPT = `You answer questions about the Bun documentation using only the context chunks in the user message.

Rules:
1. Use only facts stated in the context. Do not use prior knowledge, even if you know the answer.
2. Every context chunk starts with a header like [chunk:123]. Put the number of every chunk you used in "citations", e.g. 123. Cite only numbers that appear in a header of the context.
3. If the context does not contain the answer, reply with "found": false, "answer": "${NOT_FOUND}" and "citations": [].
4. Otherwise reply with "found": true, a concise answer and at least one citation.

Reply with only a JSON object: {"found": boolean, "answer": string, "citations": number[]}`;

/** `[chunk:123] (source_path > heading)\n<content>` per chunk, in retrieval order, separated by a blank line. */
export function formatContext(docs: Document<ChunkMetadata>[]): string {
  return docs
    .map((d) => `[chunk:${d.metadata.chunk_id}] (${sourceLabel(d.metadata)})\n${d.pageContent.trim()}`)
    .join('\n\n');
}

/** `context` is the output of formatContext, built once per request so answer() can return it too. */
export function buildAnswerMessages(question: string, context: string): Message[] {
  return [
    { role: 'system', content: ANSWER_SYSTEM_PROMPT },
    { role: 'user', content: `Context:\n\n${context}\n\nQuestion: ${question}` },
  ];
}
