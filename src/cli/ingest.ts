// bun run ingest [--chunk md-512-64] [--embed <model>] [--root corpus/bun-docs/docs] [--force] [--prune]
// Idempotent ingest of the corpus into documents, chunks and emb_<model> (docs/PLAN.md M7).
import { randomUUID } from 'node:crypto';
import { parseArgs } from 'node:util';
import { type Config, loadConfig } from '../config';
import { connect, databaseUrl } from '../db/client';
import { ensureEmbeddingTables } from '../db/embeddingTables';
import { DEFAULT_CHUNK_CONFIG, parseChunkConfigKey } from '../ingest/chunk';
import { ingest } from '../ingest/ingest';
import { DEFAULT_CORPUS_ROOT } from '../ingest/loadMarkdown';
import { createEmbeddingProvider } from '../llm/registry';
import { PostgresLlmCallSink } from '../obs/logger';
import { describeError } from './errors';

const USAGE =
  `Usage: bun run ingest [--chunk ${DEFAULT_CHUNK_CONFIG}] [--embed <model>] [--root ${DEFAULT_CORPUS_ROOT}] ` +
  '[--force] [--prune]\n' +
  "  --force  re-chunk every document for --chunk (this model's vectors are reused by chunk hash)\n" +
  '  --prune  delete documents whose file is no longer under --root';

async function main(argv: string[]): Promise<number> {
  let args: ReturnType<typeof parse>;
  try {
    args = parse(argv);
  } catch (err) {
    console.error(`${err instanceof Error ? err.message : String(err)}\n${USAGE}`);
    return 2;
  }
  if (args.help) {
    console.log(USAGE);
    return 0;
  }

  let config: Config | undefined;
  try {
    // --embed goes through the config schema, so unknown models fail with the usual message.
    config = loadConfig({ ...process.env, ...(args.embed && { EMBED_MODEL: args.embed }) });
    const chunkConfig = parseChunkConfigKey(args.chunk);
    if (!config.corpusSha) {
      throw new Error('CORPUS_SHA is not set: copy it from .env.example (the commit fetch-corpus.sh pins)');
    }
    const sql = connect(databaseUrl());
    try {
      await ensureEmbeddingTables(sql, { [config.embedModel]: config.embedding });
      const runId = `ingest-${randomUUID()}`;
      const embedder = createEmbeddingProvider(config, new PostgresLlmCallSink(sql, { runId }));
      const start = performance.now();
      const stats = await ingest({
        sql,
        root: args.root,
        chunkConfig,
        embedder,
        embedding: config.embedding,
        corpusVersion: config.corpusSha,
        force: args.force,
        prune: args.prune,
      });
      const seconds = ((performance.now() - start) / 1000).toFixed(1);
      const d = stats.documents;
      console.log(
        [
          `Ingested ${args.root} with ${chunkConfig.key} / ${config.embedModel} in ${seconds} s (run ${runId})`,
          `documents: ${d.inserted} inserted, ${d.replaced} replaced, ${d.configAdded} config added, ${d.rechunked} rechunked, ` +
            `${d.backfilled} backfilled, ${d.skipped} skipped, ${d.removed} removed, ${d.stale} stale`,
          `chunks written: ${stats.chunksWritten}; embeddings written: ${stats.embeddingsWritten}, ` +
            `computed: ${stats.embeddingsComputed} in ${stats.embedCalls} calls`,
          ...(d.stale > 0
            ? [`${d.stale} documents are no longer under ${args.root}; --prune deletes them`]
            : []),
        ].join('\n'),
      );
    } finally {
      await sql.end();
    }
    return 0;
  } catch (err) {
    console.error(
      config
        ? describeError(err, { baseURL: config.baseURL, model: config.embedModel })
        : err instanceof Error
          ? err.message
          : String(err),
    );
    return 1;
  }
}

function parse(argv: string[]) {
  const { values } = parseArgs({
    args: argv,
    strict: true,
    allowPositionals: false,
    options: {
      chunk: { type: 'string', default: DEFAULT_CHUNK_CONFIG },
      embed: { type: 'string' },
      root: { type: 'string', default: DEFAULT_CORPUS_ROOT },
      force: { type: 'boolean', default: false },
      prune: { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  return values;
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
