/**
 * Entry plumbing shared by the hook scripts (`plan-review-gate.ts`, `code-review-gate.ts`): reading the payload
 * with an optional debug dump, and running `main` fail-open.
 */
import { appendFileSync } from 'node:fs';

export interface HookStdin {
  stdin: string;
  /** Dump file from the debug env var, or `null` when unset/empty. */
  debug: string | null;
}

/** Reads the hook payload; if `debugEnv` names a file, appends the raw payload to it. */
export async function readHookInput(debugEnv: string): Promise<HookStdin> {
  const stdin = await Bun.stdin.text();
  const debugPath = process.env[debugEnv];
  const debug = debugPath !== undefined && debugPath.length > 0 ? debugPath : null;

  if (debug !== null) {
    appendFileSync(debug, `${stdin.trim()}\n`);
  }

  return { stdin, debug };
}

/**
 * Fail open: any error ends in exit 0 with no stdout and the error on stderr. A hook that exits non-zero or
 * prints half a decision would block the agent — worse than a skipped check.
 */
export async function runFailOpen(label: string, main: () => Promise<void>): Promise<void> {
  try {
    await main();
  } catch (error) {
    console.error(`[${label}]`, error);
  }
}
