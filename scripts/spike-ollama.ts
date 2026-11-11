// M1 Ollama capability spike (docs/PLAN.md M1). Local only, never run in CI.
// Checks structured output paths, tool calling, embeddings and usage fields against Ollama's
// OpenAI-compatible endpoint and prints a Markdown table for docs/SPIKE.md.
import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z } from 'zod';

export type CheckId = 'a' | 'b' | 'c' | 'd' | 'e' | 'f' | 'g';
export interface CheckResult {
  id: CheckId;
  model: string;
  pass: boolean;
  detail: string;
}

// Frozen as the M1 record (docs/SPIKE.md): these constants intentionally do not read config/models.json,
// which is the source of truth for the provider layer from M2 on.
export const GENERATOR = 'qwen2.5:7b';
export const JUDGE = 'gemma3:12b';
export const CHAT_MODELS = [GENERATOR, JUDGE] as const;
export const EMBED_MODELS = [
  { model: 'nomic-embed-text', dim: 768 },
  { model: 'mxbai-embed-large', dim: 1024 },
] as const;

const STRUCTURED_PATHS = ['b', 'c', 'g'] as const;

const passed = (results: CheckResult[], id: CheckId, model: string): boolean =>
  results.some((r) => r.id === id && r.model === model && r.pass);

/** First passing structured output path per model, in the preference order of PLAN M1 (b, c, g). */
export function chosenPath(results: CheckResult[], model: string): CheckId | undefined {
  return STRUCTURED_PATHS.find((id) => passed(results, id, model));
}

/**
 * PLAN M1 exit rule: each chat model has (b) or (c) or (g); (d) on the generator; (e) for every
 * embedding model; (f) for every chat and embedding model. Missing checks count as failed.
 */
export function exitCode(results: CheckResult[]): 0 | 1 {
  const ok =
    CHAT_MODELS.every((m) => chosenPath(results, m) !== undefined) &&
    passed(results, 'd', GENERATOR) &&
    EMBED_MODELS.every(({ model }) => passed(results, 'e', model)) &&
    [...CHAT_MODELS, ...EMBED_MODELS.map((e) => e.model)].every((m) => passed(results, 'f', m));
  return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------------------------------

const Answer = z.object({
  found: z.boolean(),
  answer: z.string(),
  citations: z.array(z.number().int()),
});

const HAND_WRITTEN_SCHEMA = {
  type: 'object',
  properties: {
    found: { type: 'boolean' },
    answer: { type: 'string' },
    citations: { type: 'array', items: { type: 'integer' } },
  },
  required: ['found', 'answer', 'citations'],
  additionalProperties: false,
};

const SYSTEM =
  'Answer the question using only the context. Cite the chunk ids you used in "citations". ' +
  'If the context does not contain the answer, set "found" to false.';
const USER =
  '[chunk:101] Environment variables from .env files are loaded automatically.\n' +
  '[chunk:102] Dependencies are installed with the install command.\n\n' +
  'Question: How are .env files handled?';

const messages = (system = SYSTEM): OpenAI.Chat.ChatCompletionMessageParam[] => [
  { role: 'system', content: system },
  { role: 'user', content: USER },
];

const SAMPLING = { temperature: 0, seed: 42 } as const;

const oneLine = (s: string, max = 160): string => {
  const flat = s.replace(/\s+/g, ' ').replaceAll('|', '/').trim();
  return flat.length > max ? `${flat.slice(0, max)}…` : flat;
};

const errorText = (err: unknown): string => oneLine(err instanceof Error ? err.message : String(err));

async function run(
  id: CheckId,
  model: string,
  fn: () => Promise<{ pass: boolean; detail: string }>,
): Promise<CheckResult> {
  const start = performance.now();
  try {
    const { pass, detail } = await fn();
    return { id, model, pass, detail: `${Math.round(performance.now() - start)} ms; ${detail}` };
  } catch (err) {
    return {
      id,
      model,
      pass: false,
      detail: `${Math.round(performance.now() - start)} ms; error: ${errorText(err)}`,
    };
  }
}

function validateJson(content: string | null | undefined): { pass: boolean; detail: string } {
  if (!content) {
    return { pass: false, detail: 'empty content' };
  }
  let json: unknown;
  try {
    json = JSON.parse(content);
  } catch {
    return { pass: false, detail: `invalid JSON: ${oneLine(content, 80)}` };
  }
  const parsed = Answer.safeParse(json);
  return parsed.success
    ? { pass: true, detail: oneLine(JSON.stringify(parsed.data), 100) }
    : { pass: false, detail: `schema: ${oneLine(parsed.error.message, 100)}` };
}

async function chatChecks(client: OpenAI, model: string): Promise<CheckResult[]> {
  const results: CheckResult[] = [];

  results.push(
    await run('a', model, async () => {
      const res = await client.chat.completions.create({
        model,
        messages: messages(),
        ...SAMPLING,
        response_format: {
          type: 'json_schema',
          json_schema: { name: 'answer', strict: true, schema: HAND_WRITTEN_SCHEMA },
        },
      });
      return validateJson(res.choices[0]?.message.content);
    }),
  );

  results.push(
    await run('b', model, async () => {
      const res = await client.chat.completions.parse({
        model,
        messages: messages(),
        ...SAMPLING,
        response_format: zodResponseFormat(Answer, 'answer'),
      });
      const parsed = res.choices[0]?.message.parsed;
      if (parsed == null) {
        return { pass: false, detail: 'message.parsed is null' };
      }
      const check = Answer.safeParse(parsed);
      return check.success
        ? { pass: true, detail: oneLine(JSON.stringify(check.data), 100) }
        : { pass: false, detail: `schema: ${oneLine(check.error.message, 100)}` };
    }),
  );

  results.push(
    await run('c', model, async () => {
      const res = await client.chat.completions.create({
        model,
        messages: messages(),
        ...SAMPLING,
        response_format: {
          type: 'json_schema',
          json_schema: { name: 'answer', strict: true, schema: z.toJSONSchema(Answer) },
        },
      });
      return validateJson(res.choices[0]?.message.content);
    }),
  );

  results.push(
    await run('g', model, async () => {
      const system = `${SYSTEM}\nRespond only with JSON matching this schema: ${JSON.stringify(z.toJSONSchema(Answer))}`;
      const res = await client.chat.completions.create({
        model,
        messages: messages(system),
        ...SAMPLING,
        response_format: { type: 'json_object' },
      });
      return validateJson(res.choices[0]?.message.content);
    }),
  );

  if (model === GENERATOR) {
    results.push(
      await run('d', model, async () => {
        const res = await client.chat.completions.create({
          model,
          ...SAMPLING,
          messages: [
            {
              role: 'system',
              content: 'Use the search_docs tool to look up documentation before answering.',
            },
            { role: 'user', content: 'How do I read environment variables in Bun?' },
          ],
          tools: [
            {
              type: 'function',
              function: {
                name: 'search_docs',
                description: 'Search the Bun documentation.',
                parameters: {
                  type: 'object',
                  properties: { query: { type: 'string' }, k: { type: 'integer' } },
                  required: ['query'],
                },
              },
            },
          ],
          tool_choice: 'auto',
        });
        const call = res.choices[0]?.message.tool_calls?.[0];
        if (call?.type !== 'function') {
          return { pass: false, detail: 'no function tool call' };
        }
        if (call.function.name !== 'search_docs') {
          return { pass: false, detail: `called ${call.function.name}` };
        }
        const args: unknown = JSON.parse(call.function.arguments);
        const ok = typeof args === 'object' && args !== null && 'query' in args;
        return { pass: ok, detail: `args ${oneLine(call.function.arguments, 100)}` };
      }),
    );
  }

  results.push(
    await run('f', model, async () => {
      const res = await client.chat.completions.create({
        model,
        ...SAMPLING,
        max_tokens: 5,
        messages: [{ role: 'user', content: 'Say ok' }],
      });
      const u = res.usage;
      const ok = typeof u?.prompt_tokens === 'number' && typeof u.completion_tokens === 'number';
      return {
        pass: ok,
        detail: u
          ? `prompt_tokens=${u.prompt_tokens}, completion_tokens=${u.completion_tokens}`
          : 'usage missing',
      };
    }),
  );

  return results;
}

async function embedChecks(client: OpenAI, model: string, dim: number): Promise<CheckResult[]> {
  let usage: OpenAI.CreateEmbeddingResponse.Usage | undefined;
  let called = false;
  const e = await run('e', model, async () => {
    const res = await client.embeddings.create({
      model,
      input: ['search_document: Bun reads .env files automatically.'],
      encoding_format: 'float',
    });
    called = true;
    usage = res.usage;
    const len = res.data[0]?.embedding.length ?? 0;
    return { pass: len === dim, detail: `${len} floats (expected ${dim})` };
  });
  const f: CheckResult = {
    id: 'f',
    model,
    pass: typeof usage?.prompt_tokens === 'number',
    detail: !called
      ? 'no response (see e)'
      : usage
        ? `prompt_tokens=${usage.prompt_tokens}`
        : 'usage missing',
  };
  return [e, f];
}

async function main(): Promise<number> {
  const baseURL = process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434/v1';
  const root = baseURL.replace(/\/v1\/?$/, '');

  let version: string;
  try {
    const res = await fetch(`${root}/api/version`, { signal: AbortSignal.timeout(5_000) });
    version = ((await res.json()) as { version?: string }).version ?? 'unknown';
  } catch (err) {
    console.error(
      `Ollama not reachable at ${root} (${errorText(err)}). Start it: brew services start ollama`,
    );
    return 2;
  }

  const client = new OpenAI({ baseURL, apiKey: 'ollama', maxRetries: 0, timeout: 120_000 });
  const results: CheckResult[] = [];
  // Grouped per model: the two chat models evict each other (docs/SPIKE.md, M0).
  for (const model of CHAT_MODELS) {
    console.error(`checking ${model} …`);
    results.push(...(await chatChecks(client, model)));
  }
  for (const { model, dim } of EMBED_MODELS) {
    console.error(`checking ${model} …`);
    results.push(...(await embedChecks(client, model, dim)));
  }

  const order: CheckId[] = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
  results.sort((x, y) => order.indexOf(x.id) - order.indexOf(y.id));

  console.log(`Ollama version: ${version}\n`);
  console.log('| Check | Model | Result | Detail |\n|---|---|---|---|');
  for (const r of results) {
    console.log(`| (${r.id}) | ${r.model} | ${r.pass ? 'pass' : 'fail'} | ${r.detail} |`);
  }
  console.log('\nChosen structured output path:');
  for (const model of CHAT_MODELS) {
    const path = chosenPath(results, model);
    console.log(`- ${model}: ${path ? `(${path})` : 'none'}`);
    if (path === 'g') {
      console.log(`  WARNING: ${model} falls back to json_object; docs/SPIKE.md must state this.`);
    }
  }

  const code = exitCode(results);
  console.log(`\nexit ${code}`);
  return code;
}

if (import.meta.main) {
  process.exit(await main());
}
