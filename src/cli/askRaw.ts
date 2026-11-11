// bun run ask:raw "question" — one raw chat call through the provider layer, no retrieval (docs/PLAN.md M2).
import { APIError } from 'openai';
import { type Config, loadConfig } from '../config';
import { createChatProvider } from '../llm/registry';
import { isConnectionError } from '../llm/retry';

export function describeError(err: unknown, config: Pick<Config, 'baseURL' | 'chatModel'>): string {
  if (isConnectionError(err)) {
    return `Ollama not reachable at ${config.baseURL}, start it (ollama serve)`;
  }
  if (err instanceof APIError) {
    const hint = err.status === 404 ? ` (is the model pulled? ollama pull ${config.chatModel})` : '';
    return `Model call failed with status ${err.status}: ${err.message}${hint}`;
  }
  return err instanceof Error ? err.message : String(err);
}

async function main(argv: string[]): Promise<number> {
  const question = argv.join(' ').trim();
  if (!question) {
    console.error('Usage: bun run ask:raw "<question>"');
    return 2;
  }
  let config: Config | undefined;
  try {
    config = loadConfig();
    const res = await createChatProvider(config).generate({
      messages: [{ role: 'user', content: question }],
    });
    console.log(res.text ?? '');
    return 0;
  } catch (err) {
    console.error(config ? describeError(err, config) : err instanceof Error ? err.message : String(err));
    return 1;
  }
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
