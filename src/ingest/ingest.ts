// Idempotent ingest (docs/PLAN.md §4, M7): documents, chunks per chunk config, embeddings per model table.
//
// Per document and chunk config one action:
// - skip:      hash equal, chunks exist and all have an embedding in this model table (or the body yields none)
// - backfill:  hash equal, chunks exist, some lack an embedding in this model table (new model): embedding rows only
// - addConfig: hash equal, no chunks for this config: insert this config's chunks, document row untouched
// - rechunk:   hash equal and force: replace this config's chunks only; other configs and their vectors stay
// - replace:   new document or changed hash: upsert the document, delete all its chunks (every config, all are
//              stale), insert the new ones
// Two phases: phase 1 reads and embeds without writing (stored vectors are read before any delete could drop
// them), phase 2 writes each document in its own transaction.
import type { Sql } from 'postgres';
import type { EmbeddingModelConfig } from '../config';
import { tableName } from '../db/embeddingTables';
import { countTokens } from '../llm/tokens';
import type { EmbeddingProvider } from '../llm/types';
import { type ChunkConfig, chunkDocument } from './chunk';
import { chunkHash, documentHash, embedText } from './hash';
import { listCorpus, loadMarkdown } from './loadMarkdown';

export const EMBED_BATCH_SIZE = 32;
/** Source paths listed in the context warning. */
const WARN_EXAMPLES = 3;

export interface IngestOptions {
  sql: Sql;
  /** Corpus root; documents.source_path is relative to it. */
  root: string;
  chunkConfig: ChunkConfig;
  /** Should log to llm_calls (registry with a Postgres sink); ingest only calls embed(…, 'document'). */
  embedder: EmbeddingProvider;
  /** Of the embedder's model: prefix and context length for the oversize warning. */
  embedding: Pick<EmbeddingModelConfig, 'documentPrefix' | 'contextTokens'>;
  corpusVersion: string;
  /**
   * Re-chunk every document for this config (e.g. after a chunker change); this model's vectors are reused by
   * chunk hash, other models' vectors of the replaced chunks are dropped and backfilled on their next run.
   */
  force?: boolean;
  /** Delete documents whose file is no longer under root. Off by default: a wrong root would wipe the rest. */
  prune?: boolean;
  /** Receives the oversize warning; defaults to console.warn. */
  warn?: (message: string) => void;
}

export interface IngestStats {
  documents: {
    inserted: number;
    replaced: number;
    configAdded: number;
    rechunked: number;
    backfilled: number;
    skipped: number;
    /** In the database but not under root, and not pruned. */
    stale: number;
    removed: number;
  };
  chunksWritten: number;
  embeddingsWritten: number;
  /** Unique embed texts sent to the model. */
  embeddingsComputed: number;
  embedCalls: number;
  /** Computed embed inputs (with model prefix) above the model's contextTokens, by the cl100k estimate. */
  oversizedInputs: number;
}

interface ChunkRow {
  /** Set for backfill (existing rows), absent for chunks still to insert. */
  id?: number;
  ordinal: number;
  headingPath: string;
  content: string;
  contentHash: string;
  tokenCount: number;
}

type Planned =
  | { action: 'skip'; sourcePath: string }
  | {
      action: 'backfill' | 'addConfig' | 'rechunk';
      sourcePath: string;
      documentId: number;
      chunks: ChunkRow[];
    }
  | {
      action: 'replace';
      documentId?: number;
      sourcePath: string;
      title: string;
      contentHash: string;
      chunks: ChunkRow[];
    };

interface DocState {
  id: number;
  contentHash: string;
  chunks: number;
  missing: number;
}

function required<T>(value: T | undefined, what: string): T {
  if (value === undefined) {
    throw new Error(`Ingest invariant broken: missing ${what}`);
  }
  return value;
}

export async function ingest(opts: IngestOptions): Promise<IngestStats> {
  const { sql, root, chunkConfig, embedder, embedding, corpusVersion } = opts;
  const warn = opts.warn ?? console.warn;
  // Before any database access: a missing or empty corpus must never reach the writes below.
  const files = await listCorpus(root);
  const table = sql(tableName(embedder.model));
  const config = chunkConfig.key;
  const stats: IngestStats = {
    documents: {
      inserted: 0,
      replaced: 0,
      configAdded: 0,
      rechunked: 0,
      backfilled: 0,
      skipped: 0,
      stale: 0,
      removed: 0,
    },
    chunksWritten: 0,
    embeddingsWritten: 0,
    embeddingsComputed: 0,
    embedCalls: 0,
    oversizedInputs: 0,
  };

  // The key determines the row; the table CHECKs validate the ranges.
  await sql`
    INSERT INTO chunk_configs (key, chunk_size, chunk_overlap, splitter)
    VALUES (${config}, ${chunkConfig.chunkSize}, ${chunkConfig.chunkOverlap}, ${chunkConfig.splitter})
    ON CONFLICT (key) DO NOTHING`;

  const stateRows = await sql<
    { id: string; source_path: string; content_hash: string; chunks: number; missing: number }[]
  >`
    SELECT d.id, d.source_path, d.content_hash,
           count(c.id)::int AS chunks,
           count(c.id) FILTER (WHERE e.chunk_id IS NULL)::int AS missing
    FROM documents d
    LEFT JOIN chunks c ON c.document_id = d.id AND c.chunk_config = ${config}
    LEFT JOIN ${table} e ON e.chunk_id = c.id
    GROUP BY d.id`;
  const state = new Map<string, DocState>(
    stateRows.map((r) => [
      r.source_path,
      { id: Number(r.id), contentHash: r.content_hash, chunks: r.chunks, missing: r.missing },
    ]),
  );

  // Phase 1: plan every document, no writes.
  const planned: Planned[] = [];
  const toChunkRows = async (doc: Parameters<typeof chunkDocument>[0]): Promise<ChunkRow[]> =>
    (await chunkDocument(doc, chunkConfig)).map((c) => ({ ...c, contentHash: chunkHash(c) }));
  for (const sourcePath of files) {
    const doc = await loadMarkdown(root, sourcePath);
    const contentHash = documentHash(doc);
    const s = state.get(sourcePath);
    if (!s || s.contentHash !== contentHash) {
      const chunks = await toChunkRows(doc);
      planned.push({
        action: 'replace',
        documentId: s?.id,
        sourcePath,
        title: doc.title,
        contentHash,
        chunks,
      });
    } else if (opts.force) {
      planned.push({
        action: 'rechunk',
        sourcePath,
        documentId: s.id,
        chunks: await toChunkRows(doc),
      });
    } else if (s.chunks === 0) {
      const chunks = await toChunkRows(doc);
      planned.push(
        chunks.length > 0
          ? { action: 'addConfig', sourcePath, documentId: s.id, chunks }
          : { action: 'skip', sourcePath },
      );
    } else if (s.missing > 0) {
      // Rebuilt from the stored rows, not by re-chunking, so the embed texts match the stored content_hash.
      const rows = await sql<
        {
          id: string;
          ordinal: number;
          heading_path: string;
          content: string;
          content_hash: string;
          token_count: number;
        }[]
      >`
        SELECT c.id, c.ordinal, c.heading_path, c.content, c.content_hash, c.token_count
        FROM chunks c LEFT JOIN ${table} e ON e.chunk_id = c.id
        WHERE c.document_id = ${s.id} AND c.chunk_config = ${config} AND e.chunk_id IS NULL
        ORDER BY c.ordinal`;
      const chunks = rows.map((r) => ({
        id: Number(r.id),
        ordinal: r.ordinal,
        headingPath: r.heading_path,
        content: r.content,
        contentHash: r.content_hash,
        tokenCount: r.token_count,
      }));
      planned.push({ action: 'backfill', sourcePath, documentId: s.id, chunks });
    } else {
      planned.push({ action: 'skip', sourcePath });
    }
  }

  // Vectors by chunk hash, as pgvector text: stored ones first, then computed for the rest.
  const needed = new Map<string, { text: string; sourcePath: string }>();
  for (const p of planned) {
    if (p.action !== 'skip') {
      for (const c of p.chunks) {
        needed.set(c.contentHash, { text: embedText(c), sourcePath: p.sourcePath });
      }
    }
  }
  const vectors = new Map<string, string>();
  if (needed.size > 0) {
    const stored = await sql<{ content_hash: string; embedding: string }[]>`
      SELECT DISTINCT ON (content_hash) content_hash, embedding::text AS embedding
      FROM ${table} WHERE content_hash = ANY(${[...needed.keys()]})`;
    for (const r of stored) {
      vectors.set(r.content_hash, r.embedding);
    }
  }
  const unresolved = [...needed].filter(([hash]) => !vectors.has(hash));
  warnOversized(unresolved, embedding, stats, warn);
  for (let i = 0; i < unresolved.length; i += EMBED_BATCH_SIZE) {
    const batch = unresolved.slice(i, i + EMBED_BATCH_SIZE);
    const res = await embedder.embed(
      batch.map(([, n]) => n.text),
      'document',
    );
    stats.embedCalls += 1;
    batch.forEach(([hash], j) => {
      // JSON array text is pgvector's input format.
      vectors.set(hash, JSON.stringify(required(res.vectors[j], `embedding ${j} of a batch`)));
    });
    stats.embeddingsComputed += batch.length;
  }
  const vectorOf = (hash: string): string => required(vectors.get(hash), `embedding for chunk hash ${hash}`);

  // Phase 2: one transaction per document.
  for (const p of planned) {
    if (p.action === 'skip') {
      stats.documents.skipped += 1;
      continue;
    }
    await sql.begin(async (tx) => {
      let documentId: number;
      let chunks: { id: number; contentHash: string }[];
      if (p.action === 'backfill') {
        documentId = p.documentId;
        chunks = p.chunks.map((c) => ({ id: required(c.id, 'stored chunk id'), contentHash: c.contentHash }));
      } else {
        if (p.action === 'replace') {
          const [doc] = await tx<{ id: string }[]>`
            INSERT INTO documents (source_path, title, content_hash, corpus_version)
            VALUES (${p.sourcePath}, ${p.title}, ${p.contentHash}, ${corpusVersion})
            ON CONFLICT (source_path) DO UPDATE SET
              title = EXCLUDED.title, content_hash = EXCLUDED.content_hash,
              corpus_version = EXCLUDED.corpus_version, ingested_at = now()
            RETURNING id`;
          if (!doc) {
            throw new Error(`Upsert of ${p.sourcePath} returned no row`);
          }
          documentId = Number(doc.id);
          // Every config: once the content changed, other configs' chunks are stale too (embeddings cascade).
          await tx`DELETE FROM chunks WHERE document_id = ${documentId}`;
        } else {
          documentId = p.documentId;
          if (p.action === 'rechunk') {
            await tx`DELETE FROM chunks WHERE document_id = ${documentId} AND chunk_config = ${config}`;
          }
        }
        chunks = [];
        if (p.chunks.length > 0) {
          const inserted = await tx<{ id: string; ordinal: number }[]>`
            INSERT INTO chunks ${tx(
              p.chunks.map((c) => ({
                document_id: documentId,
                chunk_config: config,
                ordinal: c.ordinal,
                heading_path: c.headingPath,
                content: c.content,
                content_hash: c.contentHash,
                token_count: c.tokenCount,
              })),
            )}
            RETURNING id, ordinal`;
          const idByOrdinal = new Map(inserted.map((r) => [r.ordinal, Number(r.id)]));
          chunks = p.chunks.map((c) => ({
            id: required(idByOrdinal.get(c.ordinal), `id of inserted chunk ${c.ordinal}`),
            contentHash: c.contentHash,
          }));
          stats.chunksWritten += chunks.length;
        }
      }
      if (chunks.length > 0) {
        await tx`
          INSERT INTO ${table} ${tx(
            chunks.map((c) => ({
              chunk_id: c.id,
              chunk_config: config,
              content_hash: c.contentHash,
              embedding: vectorOf(c.contentHash),
            })),
          )}`;
        stats.embeddingsWritten += chunks.length;
      }
    });
    if (p.action === 'replace') {
      stats.documents[p.documentId === undefined ? 'inserted' : 'replaced'] += 1;
    } else {
      const counter = { backfill: 'backfilled', addConfig: 'configAdded', rechunk: 'rechunked' } as const;
      stats.documents[counter[p.action]] += 1;
    }
  }

  // Rows of files present in this corpus count as verified against it, written or not.
  await sql`
    UPDATE documents SET corpus_version = ${corpusVersion}
    WHERE corpus_version <> ${corpusVersion} AND source_path = ANY(${files})`;
  const present = new Set(files);
  const gone = [...state.keys()].filter((path) => !present.has(path));
  if (opts.prune && gone.length > 0) {
    const removed = await sql`DELETE FROM documents WHERE source_path = ANY(${gone})`;
    stats.documents.removed = removed.count;
  } else {
    stats.documents.stale = gone.length;
  }
  return stats;
}

/** One line for all computed inputs above the model context; the cl100k count only approximates the model's. */
function warnOversized(
  unresolved: [string, { text: string; sourcePath: string }][],
  embedding: IngestOptions['embedding'],
  stats: IngestStats,
  warn: (message: string) => void,
): void {
  const over: { tokens: number; sourcePath: string }[] = [];
  for (const [, n] of unresolved) {
    const tokens = countTokens(embedding.documentPrefix + n.text);
    if (tokens > embedding.contextTokens) {
      over.push({ tokens, sourcePath: n.sourcePath });
    }
  }
  stats.oversizedInputs = over.length;
  if (over.length > 0) {
    const max = Math.max(...over.map((o) => o.tokens));
    const examples = [...new Set(over.map((o) => o.sourcePath))].slice(0, WARN_EXAMPLES).join(', ');
    warn(
      `${over.length} embedding inputs exceed the model context of ${embedding.contextTokens} tokens ` +
        `(max ${max}, cl100k estimate) and may be truncated, e.g. ${examples}`,
    );
  }
}
