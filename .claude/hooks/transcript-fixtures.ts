/**
 * Transcript line builders shared by the hook tests (`plan-review-gate.test.ts`, `code-review-gate.test.ts`).
 * Each returns one JSONL line as the session transcript stores it.
 */
import { type TranscriptLine, parseTranscript } from './plan-review-transcript';

export function userTurn(text = 'build me this'): string {
  return JSON.stringify({ type: 'user', message: { content: [{ type: 'text', text }] } });
}

export function agentCall(subagentType: string, timestamp?: string): string {
  return JSON.stringify({
    type: 'assistant',
    timestamp,
    message: {
      content: [{ type: 'tool_use', name: 'Agent', input: { subagent_type: subagentType } }],
    },
  });
}

/** Written by the harness when a background subagent finishes — typed `user`, but not a user turn. */
export function taskNotification(): string {
  return JSON.stringify({
    type: 'user',
    origin: { kind: 'task-notification', producer: 'session-task' },
    message: { content: '<task-notification>done</task-notification>' },
  });
}

export function transcript(...lines: string[]): TranscriptLine[] {
  return parseTranscript(lines.join('\n'));
}
