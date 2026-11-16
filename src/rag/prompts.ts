// Answer prompt: NOT_FOUND rule and citation format (docs/PLAN.md M9); query rewrite (M15), expansion (M16),
// rerank (M17) and agent (M18) prompts.
import type { Document } from '@langchain/core/documents';
import type { Message } from '../llm/types';
import { type ChunkMetadata, sourceLabel } from '../retrieval/pgRetriever';
import type { SearchHit } from '../retrieval/search';
import { MAX_EXPANSIONS, MAX_RERANK_SCORE, MAX_TOOL_CALLS, RERANK_CONTEXT_CHARS } from './pipeline';

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

/** Whitespace collapsed, cut to `max` characters with an ellipsis (retrieve previews, rerank truncation). */
export function preview(text: string, max: number): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat;
}

/** `[chunk:123] (source_path > heading)\n<text>`: the header every prompt that shows chunks uses. */
export const formatChunk = (
  meta: Pick<ChunkMetadata, 'chunk_id' | 'source_path' | 'heading_path'>,
  text: string,
) => `[chunk:${meta.chunk_id}] (${sourceLabel(meta)})\n${text}`;

/** formatChunk per chunk, in retrieval order, separated by a blank line. */
export function formatContext(docs: Document<ChunkMetadata>[]): string {
  return docs.map((d) => formatChunk(d.metadata, d.pageContent.trim())).join('\n\n');
}

/** `context` is the output of formatContext, built once per request so answer() can return it too. */
export function buildAnswerMessages(question: string, context: string): Message[] {
  return [
    { role: 'system', content: ANSWER_SYSTEM_PROMPT },
    { role: 'user', content: `Context:\n\n${context}\n\nQuestion: ${question}` },
  ];
}

/** Fixed like the answer settings, so a rewrite run is reproducible. */
export const REWRITE_TEMPERATURE = 0;
export const REWRITE_SEED = 42;

export const REWRITE_SYSTEM_PROMPT = `You rewrite a user question about Bun (the JavaScript runtime, package manager, bundler and test runner) into one search query for the Bun documentation.

Rules:
1. Keep every identifier, API name, command, flag, file name, config key and error message from the question exactly as written.
2. You may make vague phrasing concrete, expand abbreviations and add the terms the Bun documentation would use for the topic (for example "bun install" for installing packages).
3. Do not answer the question. Do not add API names, flags or facts that the question does not imply.
4. Write one line in English, as a standalone query.

Reply with only a JSON object: {"rewritten": string}`;

export function buildRewriteMessages(question: string): Message[] {
  return [
    { role: 'system', content: REWRITE_SYSTEM_PROMPT },
    { role: 'user', content: `Question: ${question}` },
  ];
}

/** Fixed like the answer settings, so an expansion run is reproducible (M16). */
export const EXPAND_TEMPERATURE = 0;
export const EXPAND_SEED = 42;

export const EXPAND_SYSTEM_PROMPT = `You write additional search queries for a user question about Bun (the JavaScript runtime, package manager, bundler and test runner). The queries search the Bun documentation together with the original question, so each one should find relevant passages the question alone might miss.

Rules:
1. Write between 1 and ${MAX_EXPANSIONS} queries. Each query is one line in English.
2. Keep every identifier, API name, command, flag, file name, config key and error message from the question exactly as written.
3. You may rephrase the question, use the terms the Bun documentation would use for the topic (for example "bun install" for installing packages), or split a question that asks several things into one query per part.
4. Do not answer the question. Do not add API names, flags or facts that the question does not imply. Do not repeat the question unchanged.

Reply with only a JSON object: {"queries": string[]}`;

export function buildExpandMessages(question: string): Message[] {
  return [
    { role: 'system', content: EXPAND_SYSTEM_PROMPT },
    { role: 'user', content: `Question: ${question}` },
  ];
}

/** Fixed like the answer settings, so a rerank run is reproducible (M17). */
export const RERANK_TEMPERATURE = 0;
export const RERANK_SEED = 42;

export const RERANK_SYSTEM_PROMPT = `You rank excerpts from the Bun documentation (the JavaScript runtime, package manager, bundler and test runner) by how relevant they are to a user question.

Rules:
1. Every excerpt starts with a header like [chunk:123]. Give each excerpt a score from 0 to ${MAX_RERANK_SCORE}: ${MAX_RERANK_SCORE} if it states facts needed to answer the question, 0 if it is unrelated to it.
2. Judge only whether the excerpt helps answer the question. Do not judge writing quality, length or completeness, and do not answer the question.
3. Excerpts are cut off; judge what is shown.
4. List every excerpt exactly once, using only chunk numbers from the headers, best first.

Reply with only a JSON object: {"ranking": [{"chunk_id": number, "score": number}]}`;

/** A rerank candidate: a search hit without its search score. */
export type RerankCandidate = Omit<SearchHit, 'score'>;

/** Characters of content per candidate: RERANK_CONTEXT_CHARS split evenly. */
export const rerankChunkChars = (candidates: number): number =>
  Math.floor(RERANK_CONTEXT_CHARS / Math.max(candidates, 1));

/** Candidates in search order, each cut to rerankChunkChars, then the question (as in the answer prompt). */
export function buildRerankMessages(question: string, candidates: readonly RerankCandidate[]): Message[] {
  const max = rerankChunkChars(candidates.length);
  const excerpts = candidates
    .map((c) =>
      formatChunk(
        { chunk_id: c.chunkId, source_path: c.sourcePath, heading_path: c.headingPath },
        preview(c.content, max),
      ),
    )
    .join('\n\n');
  return [
    { role: 'system', content: RERANK_SYSTEM_PROMPT },
    { role: 'user', content: `Excerpts:\n\n${excerpts}\n\nQuestion: ${question}` },
  ];
}

/** Fixed like the answer settings, so an agent run is reproducible (M18). */
export const AGENT_TEMPERATURE = 0;
export const AGENT_SEED = 42;

export const AGENT_SYSTEM_PROMPT = `You answer questions about the Bun documentation (the JavaScript runtime, package manager, bundler and test runner). You cannot see the documentation; search it with the search_docs tool.

Rules:
1. Call search_docs with a focused query in English. You may search at most ${MAX_TOOL_CALLS} times in total; search again with different terms only if the results so far do not answer the question.
2. Use only facts stated in the search results. Do not use prior knowledge, even if you know the answer.
3. Every result chunk starts with a header like [chunk:123]. Put the number of every chunk you used in "citations", e.g. 123. Cite only numbers that appear in a header of a search result.
4. If the search results do not contain the answer, reply with "found": false, "answer": "${NOT_FOUND}" and "citations": [].
5. When you are done searching, reply without a tool call, with only a JSON object: {"found": boolean, "answer": string, "citations": number[]}`;

export function buildAgentMessages(question: string): Message[] {
  return [
    { role: 'system', content: AGENT_SYSTEM_PROMPT },
    { role: 'user', content: `Question: ${question}` },
  ];
}
