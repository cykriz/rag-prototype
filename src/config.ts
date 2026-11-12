// Single source of configuration (docs/PLAN.md M2): env vars validated with Zod, models from config/models.json.
import { z } from 'zod';
import modelsJson from '../config/models.json';

const EmbeddingModel = z.object({
  dim: z.number().int().positive(),
  documentPrefix: z.string(),
  queryPrefix: z.string(),
});
const Models = z.object({
  profiles: z.record(z.string(), z.object({ chat: z.string(), judge: z.string() })),
  embeddings: z.record(z.string(), EmbeddingModel),
});
export type EmbeddingModelConfig = z.infer<typeof EmbeddingModel>;

export const models = Models.parse(modelsJson);

const Provider = z.enum(['ollama', 'fake']);
export type ProviderName = z.infer<typeof Provider>;

const Env = z.object({
  CHAT_PROVIDER: Provider.default('ollama'),
  EMBED_PROVIDER: Provider.default('ollama'),
  OLLAMA_BASE_URL: z.url().default('http://localhost:11434/v1'),
  MODEL_PROFILE: z.enum(Object.keys(models.profiles) as [string, ...string[]]).default('default'),
  EMBED_MODEL: z.enum(Object.keys(models.embeddings) as [string, ...string[]]).default('nomic-embed-text'),
  DATABASE_URL: z.url().optional(),
});

export interface Config {
  chatProvider: ProviderName;
  embedProvider: ProviderName;
  baseURL: string;
  chatModel: string;
  judgeModel: string;
  embedModel: string;
  embedding: EmbeddingModelConfig;
  /** No default: database code fails fast when it is unset (src/db/client.ts). */
  databaseUrl?: string;
}

export function loadConfig(env: Record<string, string | undefined> = process.env): Config {
  // Empty strings (e.g. `CHAT_PROVIDER=` in .env) count as unset, so defaults apply.
  const cleaned = Object.fromEntries(Object.entries(env).filter(([, v]) => v !== undefined && v !== ''));
  const parsed = Env.safeParse(cleaned);
  if (!parsed.success) {
    const issues = parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ');
    throw new Error(`Invalid configuration: ${issues}`);
  }
  const e = parsed.data;
  const profile = models.profiles[e.MODEL_PROFILE];
  const embedding = models.embeddings[e.EMBED_MODEL];
  if (!profile || !embedding) {
    throw new Error('Invalid configuration: model profile or embedding model missing');
  }
  return {
    chatProvider: e.CHAT_PROVIDER,
    embedProvider: e.EMBED_PROVIDER,
    baseURL: e.OLLAMA_BASE_URL,
    chatModel: profile.chat,
    judgeModel: profile.judge,
    embedModel: e.EMBED_MODEL,
    embedding,
    databaseUrl: e.DATABASE_URL,
  };
}
