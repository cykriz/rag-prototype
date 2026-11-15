// Model digests from Ollama's native API (GET /api/tags), stored in result files (docs/PLAN.md §7) so a re-pulled
// model with the same tag is distinguishable.
import { z } from 'zod';

/** /api/tags only lists local models; an Ollama that accepts but never answers must not block the eval. */
const TAGS_TIMEOUT_MS = 10_000;

const Tags = z.object({ models: z.array(z.object({ name: z.string(), digest: z.string() })) });

export interface ModelDigest {
  /** Tag as Ollama lists it, e.g. `nomic-embed-text:latest`. */
  tag: string;
  digest: string;
}

/** `http://localhost:11434/v1` → `http://localhost:11434`: /api/tags is not part of the OpenAI-compatible API. */
export function ollamaApiBase(baseURL: string): string {
  return baseURL.replace(/\/+$/, '').replace(/\/v1$/, '');
}

/** Exact name first, then `<name>:latest`; undefined when the model is not pulled. */
export function findDigest(
  listed: readonly { name: string; digest: string }[],
  model: string,
): ModelDigest | undefined {
  const hit =
    listed.find((m) => m.name === model) ??
    (model.includes(':') ? undefined : listed.find((m) => m.name === `${model}:latest`));
  return hit && { tag: hit.name, digest: hit.digest };
}

/**
 * Digest of `model`. A refused connection propagates as the fetch error (describeError turns it into the
 * "Ollama not reachable" hint); a model that is not pulled throws with the pull command.
 */
export async function fetchModelDigest(
  baseURL: string,
  model: string,
  fetchFn: typeof fetch = fetch,
): Promise<ModelDigest> {
  const url = `${ollamaApiBase(baseURL)}/api/tags`;
  const res = await fetchFn(url, { signal: AbortSignal.timeout(TAGS_TIMEOUT_MS) });
  if (!res.ok) {
    throw new Error(`GET ${url} failed with status ${res.status}`);
  }
  const { models } = Tags.parse(await res.json());
  const found = findDigest(models, model);
  if (!found) {
    throw new Error(`Model ${model} is not pulled in Ollama: ollama pull ${model}`);
  }
  return found;
}
