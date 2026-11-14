// bun run ask:raw "question" — one raw chat call through the provider layer, no retrieval (docs/PLAN.md M2).
import { type Config, loadConfig } from '../config';
import { createChatProvider } from '../llm/registry';
import { describeError } from './errors';

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
    console.error(
      config
        ? describeError(err, { baseURL: config.baseURL, model: config.chatModel })
        : err instanceof Error
          ? err.message
          : String(err),
    );
    return 1;
  }
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
