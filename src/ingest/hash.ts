// Content hashes for idempotent ingest (docs/PLAN.md §3, §4): sha256 via node:crypto, hex encoded (char(64)).
import { createHash } from 'node:crypto';

export function sha256(data: string | Uint8Array): string {
  return createHash('sha256').update(data).digest('hex');
}

/**
 * documents.content_hash: the parsed title, body and description, not the raw file. Everything stored derives from
 * these, so other frontmatter edits change nothing, while a loader change that alters the body does. Without a
 * description the hash is that of title and body alone, as before M19b.
 */
export function documentHash(doc: { title: string; body: string; description?: string }): string {
  const base = `${doc.title}\0${doc.body}`;
  return sha256(doc.description === undefined ? base : `${base}\0${doc.description}`);
}

/**
 * The text sent to the embedding model, before the model's task prefix (added by the adapter per `kind`): the
 * heading path as context line, then the content (docs/PLAN.md risk table).
 */
export function embedText(chunk: { headingPath: string; content: string }): string {
  return chunk.headingPath ? `${chunk.headingPath}\n\n${chunk.content}` : chunk.content;
}

/**
 * chunks.content_hash and emb_<model>.content_hash: the embed text, so equal hashes mean equal embedding input
 * and a stored vector can be reused. The model prefix is constant per embeddings table and is not hashed.
 */
export function chunkHash(chunk: { headingPath: string; content: string }): string {
  return sha256(embedText(chunk));
}
