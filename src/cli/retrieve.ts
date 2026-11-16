// bun run retrieve "<question>" [--k 5] [--chunk md-512-64] [--embed <model>] [--search vector|fts|hybrid] [--rewrite]
// Prints the top k chunks with score, chunk ID and source (docs/PLAN.md M8, M14); with --rewrite it searches with the
// rewritten query (M15) and prints that query on stderr.
import { randomUUID } from 'node:crypto';
import type { Document } from '@langchain/core/documents';
import { createChatProvider, createEmbeddingProvider } from '../llm/registry';
import { PostgresLlmCallSink } from '../obs/logger';
import { rewriteQuery } from '../rag/rewrite';
import { type ChunkMetadata, PgRetriever, sourceLabel } from '../retrieval/pgRetriever';
import type { SearchMode } from '../retrieval/search';
import { QUESTION_FLAGS, runQuestionCli } from './common';
import { retrievalErrorMessage } from './errors';

const PREVIEW_CHARS = 120;

const USAGE = `Usage: bun run retrieve "<question>" ${QUESTION_FLAGS}`;

/** Whitespace collapsed, cut to `max` characters with an ellipsis. */
export function preview(text: string, max = PREVIEW_CHARS): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat;
}

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
    lines.push(`${' '.repeat(width + 1)}  ${preview(d.pageContent)}`);
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

function main(argv: string[]): Promise<number> {
  return runQuestionCli(argv, USAGE, async ({ args, config, chunkConfig, sql }) => {
    const { search, rewrite } = args.pipeline;
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
      const retriever = new PgRetriever({ sql, embedder, chunkConfig, k: args.k, search });
      const docs = await retriever.invoke(query);
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
