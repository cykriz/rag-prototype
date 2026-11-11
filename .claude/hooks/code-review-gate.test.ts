import { afterAll, describe, expect, test } from 'bun:test';

import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  changedPaths,
  decide,
  isDocOnly,
  parsePorcelain,
  recordSnapshot,
  snapshot,
} from './code-review-gate';
import { agentCall, taskNotification, transcript, userTurn } from './transcript-fixtures';

const CODE = ['src/a.ts'];

describe('decide', () => {
  test('blocks a code change without a review and names the changed files', () => {
    const decision = decide({}, transcript(userTurn()), ['src/a.ts', 'README.md']);

    expect(decision.kind).toBe('block');
    expect(decision.kind === 'block' && decision.reason).toContain('src/a.ts, README.md');
  });

  test('passes once review-changes has started since the last user turn', () => {
    expect(decide({}, transcript(userTurn(), agentCall('review-changes')), CODE).kind).toBe('skip');
  });

  test('a background review still counts after its task notification arrives', () => {
    const lines = transcript(userTurn(), agentCall('review-changes'), taskNotification());

    expect(decide({}, lines, CODE).kind).toBe('skip');
  });

  test('a review before the last user turn does not count', () => {
    const lines = transcript(userTurn(), agentCall('review-changes'), userTurn('next task'));

    expect(decide({}, lines, CODE).kind).toBe('block');
  });

  test('another subagent is not a review', () => {
    expect(decide({}, transcript(userTurn(), agentCall('plan-reviewer')), CODE).kind).toBe('block');
  });

  test('passes on stop_hook_active, in a subagent, without a change, without a comparison, on doc-only', () => {
    const lines = transcript(userTurn());

    expect(decide({ stop_hook_active: true }, lines, CODE).kind).toBe('skip');
    expect(decide({ agent_id: 'x' }, lines, CODE).kind).toBe('skip');
    expect(decide({}, lines, []).kind).toBe('skip');
    expect(decide({}, lines, null).kind).toBe('skip');
    expect(decide({}, lines, ['README.md', 'docs/PLAN.md']).kind).toBe('skip');
  });
});

describe('isDocOnly', () => {
  test('plain docs are doc-only; agent rule files and code are not', () => {
    expect(isDocOnly(['README.md'])).toBe(true);
    expect(isDocOnly(['.claude/commands/review.md'])).toBe(false);
    expect(isDocOnly(['CLAUDE.md'])).toBe(false);
    expect(isDocOnly(['sub/CLAUDE.md'])).toBe(false);
    expect(isDocOnly(['README.md', 'src/a.ts'])).toBe(false);
  });
});

describe('parsePorcelain', () => {
  test('reads plain entries and both sides of a rename', () => {
    expect(parsePorcelain(' M a.ts\0?? b/c.ts\0R  new.ts\0old.ts\0')).toEqual([
      'a.ts',
      'b/c.ts',
      'new.ts',
      'old.ts',
    ]);
  });
});

describe('snapshot', () => {
  const dir = mkdtempSync(join(tmpdir(), 'code-review-gate-'));
  const run = (...args: string[]) => spawnSync('git', args, { cwd: dir });

  run('init', '-q');
  writeFileSync(join(dir, 'tracked.ts'), 'export const a = 1;\n');
  run('add', 'tracked.ts');
  run('-c', 'user.name=test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'init');

  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test('is stable when nothing changes and detects tracked, new and nested untracked edits', () => {
    const clean = snapshot(dir);

    expect(clean).toEqual({});
    expect(snapshot(dir)).toEqual(clean);

    writeFileSync(join(dir, 'tracked.ts'), 'export const a = 2;\n');
    const edited = snapshot(dir);

    expect(edited).not.toBeNull();
    expect(changedPaths(clean ?? {}, edited ?? {})).toEqual(['tracked.ts']);

    mkdirSync(join(dir, 'nested', 'deep'), { recursive: true });
    writeFileSync(join(dir, 'nested', 'deep', 'new.ts'), 'one\n');
    const added = snapshot(dir) ?? {};

    expect(changedPaths(edited ?? {}, added)).toEqual(['nested/deep/new.ts']);

    writeFileSync(join(dir, 'nested', 'deep', 'new.ts'), 'two\n');

    expect(changedPaths(added, snapshot(dir) ?? {})).toEqual(['nested/deep/new.ts']);
  });

  test('skips ignored bulk directories', () => {
    const before = snapshot(dir) ?? {};
    mkdirSync(join(dir, 'corpus'), { recursive: true });
    writeFileSync(join(dir, 'corpus', 'page.md'), 'bulk\n');

    expect(changedPaths(before, snapshot(dir) ?? {})).toEqual([]);
  });

  test('returns null above the file or byte limit', () => {
    writeFileSync(join(dir, 'limit.ts'), 'dirty\n');

    expect(snapshot(dir, { maxFiles: 0, maxBytes: 1_000_000 })).toBeNull();
    expect(snapshot(dir, { maxFiles: 100, maxBytes: 1 })).toBeNull();
  });

  test('script: UserPromptSubmit snapshots silently, Stop blocks a code change, fails open without snapshot', () => {
    const script = join(import.meta.dirname, 'code-review-gate.ts');
    const session = `test-${process.pid}`;
    const transcriptFile = join(dir, '..', `${session}.jsonl`);
    const hook = (payload: object) =>
      spawnSync('bun', [script], {
        input: JSON.stringify({ session_id: session, transcript_path: transcriptFile, ...payload }),
        env: { ...process.env, CLAUDE_PROJECT_DIR: dir },
        encoding: 'utf8',
      });

    writeFileSync(transcriptFile, `${userTurn()}\n`);

    const missing = hook({ hook_event_name: 'Stop' });
    expect(missing.status).toBe(0);
    expect(missing.stdout).toBe('');

    const submit = hook({ hook_event_name: 'UserPromptSubmit' });
    expect(submit.status).toBe(0);
    expect(submit.stdout).toBe('');

    writeFileSync(join(dir, 'script-edit.ts'), 'changed\n');
    const stop = hook({ hook_event_name: 'Stop' });

    expect(stop.status).toBe(0);
    expect(JSON.parse(stop.stdout)).toEqual({
      decision: 'block',
      reason: expect.stringContaining('Files changed this turn: script-edit.ts'),
    });

    rmSync(transcriptFile, { force: true });
    rmSync(join(tmpdir(), 'claude-code-review', session), { force: true });
  });

  test('removes a stale snapshot when a new one cannot be written', () => {
    const file = join(dir, '..', `${dir.split('/').at(-1)}-snapshot`);
    writeFileSync(file, '{"old.ts":"x"}');

    recordSnapshot(dir, file, { maxFiles: 0, maxBytes: 1 });

    expect(existsSync(file)).toBe(false);
  });
});
