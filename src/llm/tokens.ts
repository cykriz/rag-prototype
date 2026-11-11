// Token counts (docs/PLAN.md M3): provider usage first, js-tiktoken cl100k_base estimate as fallback.
import { getEncoding, type Tiktoken } from 'js-tiktoken';
import type { Message, ToolCall, Usage } from './types';

export type TokenSource = 'usage' | 'tiktoken_estimate';

export interface TokenCount extends Usage {
  tokenSource: TokenSource;
}

let encoder: Tiktoken | undefined;

/** cl100k_base is not the native tokenizer of local models, so this is an estimate only. */
export function countTokens(text: string): number {
  encoder ??= getEncoding('cl100k_base');
  // Special-token strings such as <|endoftext|> count as plain text; the default would throw on them.
  return encoder.encode(text, [], []).length;
}

const toolCallTokens = (calls: ToolCall[] | undefined): number =>
  (calls ?? []).reduce((n, c) => n + countTokens(c.name) + countTokens(c.arguments), 0);

/** Message contents plus tool call names and arguments; no per-message overhead. */
export function countMessageTokens(messages: Message[]): number {
  return messages.reduce(
    (n, m) => n + countTokens(m.content ?? '') + (m.role === 'assistant' ? toolCallTokens(m.toolCalls) : 0),
    0,
  );
}

export function countOutputTokens(text: string | null, toolCalls?: ToolCall[]): number {
  return countTokens(text ?? '') + toolCallTokens(toolCalls);
}

export function resolveTokens(usage: Usage | undefined, estimate: () => Usage): TokenCount {
  return usage ? { ...usage, tokenSource: 'usage' } : { ...estimate(), tokenSource: 'tiktoken_estimate' };
}
