// bun run retrieve "<question>" [--k 5] [--chunk md-1024-128] [--embed <model>] [--search vector|fts|hybrid]
//   [--rewrite | --expand] [--rerank]
// Prints the top k chunks with score, chunk ID and source (docs/PLAN.md M8, M14); with --rewrite it searches with the
// rewritten query (M15), with --expand with the question and its expansions fused by RRF (M16), and prints the
// queries on stderr. With --rerank it fetches max(k, 20) candidates, reranks them with the chat model and prints the
// top k in rerank order (M17; the score column stays the search score), the rerank summary on stderr.
import { randomUUID } from 'node:crypto';
import type { Document } from '@langchain/core/documents';
import { createChatProvider, createEmbeddingProvider } from '../llm/registry';
import { PostgresLlmCallSink } from '../obs/logger';
import { type ExpandTrace, expandQuery, searchQueries } from '../rag/expand';
import { retrievalDepth } from '../rag/pipeline';
import { preview } from '../rag/prompts';
import { documentCandidate, type RerankTrace, rerankerFor, rerankItems } from '../rag/rerank';
import { rewriteQuery } from '../rag/rewrite';
import { type ChunkMetadata, PgRetriever, sourceLabel } from '../retrieval/pgRetriever';
import type { SearchMode } from '../retrieval/search';
import { QUESTION_FLAGS, runQuestionCli } from './common';
import { retrievalErrorMessage } from './errors';

const PREVIEW_CHARS = 120;

const USAGE = `Usage: bun run retrieve "<question>" ${QUESTION_FLAGS}`;

export function formatHits(
  question: string,
  meta: { chunkConfig: string; model: string; k: number; search: SearchMode },
  docs: Document<ChunkMetadata>[],
): string {
  const lines = [`Top ${meta.k} for "${question}" (${meta.chunkConfig} / ${meta.model}, ${meta.search})`];
  const width = String(docs.length).length;
  docs.forEach((d, i) => {
    const { chunk_id, score } = d.metadata;
    const rank = String(i + 1).padStart(width + 1);
    lines.push(`${rank}  ${score.toFixed(4)}  #${chunk_id}  ${sourceLabel(d.metadata)}`);
    lines.push(`${' '.repeat(width + 1)}  ${preview(d.pageContent, PREVIEW_CHARS)}`);
  });
  if (docs.length < meta.k) {
    lines.push(
      meta.search === 'fts'
        ? `only ${docs.length} matching chunks in ${meta.chunkConfig}`
        : `only ${docs.length} chunks in ${meta.chunkConfig}`,
    );
  }
  return lines.join('\n');
}

/** The stderr line(s) of --expand: the expansions searched besides the question, or why there are none. */
export function formatExpansion(trace: ExpandTrace): string {
  if (trace.fallback) {
    return `expand: invalid output after ${trace.repairs} repairs, searching with the question only`;
  }
  const dropped =
    trace.dropped > 0 ? ` (${trace.dropped} duplicate${trace.dropped === 1 ? '' : 's'} dropped)` : '';
  if (trace.queries.length === 0) {
    return `expand: no new queries${dropped}, searching with the question only`;
  }
  return [
    `expand: ${trace.queries.length} queries besides the question${dropped}`,
    ...trace.queries.map((q) => `  ${q}`),
  ].join('\n');
}

/** The stderr line(s) of --rerank: the kept chunks with model score and search rank, or why the order is unchanged. */
export function formatRerank(trace: RerankTrace, k: number): string {
  if (trace.fallback) {
    return `rerank: invalid output after ${trace.repairs} repairs, keeping the search order`;
  }
  if (trace.candidates.length < 2) {
    return `rerank: ${trace.candidates.length} candidate${trace.candidates.length === 1 ? '' : 's'}, nothing to rerank`;
  }
  const ignored = [
    trace.unknown > 0 && `${trace.unknown} unknown`,
    trace.duplicates > 0 && `${trace.duplicates} duplicate`,
    trace.missing > 0 && `${trace.missing} missing`,
  ].filter(Boolean);
  const kept = Math.min(k, trace.order.length);
  return [
    `rerank: ${trace.candidates.length} candidates, top ${kept} kept` +
      (ignored.length > 0 ? ` (IDs: ${ignored.join(', ')})` : ''),
    ...trace.order.slice(0, kept).map((id, i) => {
      const score = trace.scores[i];
      return `  #${id}  score ${score === null || score === undefined ? '-' : score}  search rank ${trace.candidates.indexOf(id) + 1}`;
    }),
  ].join('\n');
}

function main(argv: string[]): Promise<number> {
  return runQuestionCli(argv, USAGE, async ({ args, config, chunkConfig, sql }) => {
    const { search, rewrite, expand, rerank } = args.pipeline;
    try {
      const sink = new PostgresLlmCallSink(sql, { runId: `retrieve-${randomUUID()}` });
      let query = args.question;
      if (rewrite === 'on') {
        const trace = await rewriteQuery(createChatProvider(config, sink), args.question);
        query = trace.query;
        console.error(
          trace.fallback
            ? `rewrite: invalid output after ${trace.repairs} repairs, searching with the question`
            : `rewrite: ${query}`,
        );
      }
      const embedder = createEmbeddingProvider(config, sink);
      const k = retrievalDepth(args.pipeline, args.k);
      const retriever = new PgRetriever({ sql, embedder, chunkConfig, k, search });
      let docs: Document<ChunkMetadata>[];
      if (expand === 'on') {
        const trace = await expandQuery(createChatProvider(config, sink), args.question);
        console.error(formatExpansion(trace));
        docs = await retriever.invokeMany(searchQueries(args.question, trace));
      } else {
        docs = await retriever.invoke(query);
      }
      if (rerank === 'on') {
        const { kept, trace } = await rerankItems(
          rerankerFor(createChatProvider(config, sink)),
          args.question,
          docs,
          documentCandidate,
          args.k,
        );
        console.error(formatRerank(trace, args.k));
        docs = kept;
      }
      console.log(
        formatHits(args.question, { chunkConfig, model: config.embedModel, k: args.k, search }, docs),
      );
      return 0;
    } catch (err) {
      console.error(await retrievalErrorMessage(err, config, sql));
      return 1;
    }
  });
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
