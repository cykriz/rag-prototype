import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterAll, describe, expect, test } from 'vitest';

import { commitMessageBody, formatFindings, isBinary, parseDenylist, scanText } from './privacy-scan';

// Every value that must be flagged is assembled at runtime, so this file never trips the gate it tests.
// All of it is synthetic.
const AT = '@';
const FAKE_EMAIL = ['jane.roe', 'fictional-widgets.de'].join(AT);
const FAKE_HOME = ['', 'Users', 'jroe', 'project'].join('/');
const FAKE_LINUX_HOME = ['', 'home', 'jroe'].join('/');
const FAKE_WIN_HOME = ['C:', 'Users', 'jroe'].join('\\');
const FAKE_IP = [10, 20, 30, 40].join('.');
const FAKE_TOKENS = [
  `${'sk'}-${'a'.repeat(24)}`,
  `${'gh'}p_${'b'.repeat(36)}`,
  `${'xo'}xb-${'1'.repeat(12)}`,
  `${'AK'}IA${'C'.repeat(16)}`,
  `${'ey'}J${'d'.repeat(12)}.${'ey'}J${'e'.repeat(12)}.${'f'.repeat(12)}`,
  `${'Bear'}er ${'g'.repeat(30)}`,
  `-----BEGIN ${'PRIVATE'} KEY-----`,
  `api_key = "${'h'.repeat(16)}"`,
];

function kinds(text: string, denylist = ''): string[] {
  return scanText('f.ts', text, parseDenylist(denylist)).map((f) => f.kind);
}

describe('scanText', () => {
  test('flags an email address', () => {
    expect(kinds(`contact ${FAKE_EMAIL}`)).toEqual(['email']);
  });

  test.each([FAKE_HOME, FAKE_LINUX_HOME, FAKE_WIN_HOME])('flags a home path (%#)', (path) => {
    expect(kinds(`cd ${path}`)).toEqual(['home-path']);
  });

  test('flags a non-documentation IPv4 address', () => {
    expect(kinds(`host=${FAKE_IP}`)).toEqual(['ipv4']);
  });

  test.each(FAKE_TOKENS)('flags token pattern %#', (token) => {
    expect(kinds(`x ${token} y`)).toEqual(['token']);
  });

  test('flags a denylist entry case-insensitively', () => {
    expect(kinds('Built for FICTIONAL Widgets', 'fictional widgets')).toEqual(['denylist']);
  });

  test.each([
    ['fictional*gmbh', 'Fictional Widgets GmbH'],
    ['*.corp.invalid', 'see wiki.corp.invalid/page'],
    ['j?ne roe', 'by Jane Roe'],
  ])('matches wildcard entry %s', (entry, text) => {
    expect(kinds(text, entry)).toEqual(['denylist']);
  });

  test.each([
    ['j?ne roe', 'by Jaane Roe'],
    ['a.b', 'axb'],
    ['(x)+', 'xx'],
  ])('treats everything but * and ? literally: %s', (entry, text) => {
    expect(kinds(text, entry)).toEqual([]);
  });

  test('reports the line number, one finding per line and kind', () => {
    const findings = scanText('a.md', `clean\n${FAKE_EMAIL} ${FAKE_EMAIL}`, []);

    expect(findings).toEqual([{ file: 'a.md', line: 2, kind: 'email' }]);
  });

  test.each([
    'mail jane@example.com or ops@example.org',
    'user@corp.test and x@y.invalid',
    'git@github.com:oven-sh/bun.git',
    'http://localhost:11434/v1 and 127.0.0.1 and 0.0.0.0',
    'docs use 192.0.2.10, 198.51.100.7 and 203.0.113.99',
    'bun add zod@4.6.7 vitest@5.1.0',
    'version 999.1.2.3 is not an address',
    '~/.claude/plans/x.md and /plans/my-plan.md',
    'apiKey: "ollama"',
  ])('allows synthetic or documentation value: %s', (text) => {
    expect(kinds(text)).toEqual([]);
  });
});

describe('parseDenylist', () => {
  test('ignores comments, blanks and wildcard-only entries', () => {
    const patterns = parseDenylist('# header\n\n  Fictional Widgets \n*\n?*\nJane Roe\n');

    expect(patterns).toHaveLength(2);
  });
});

describe('commitMessageBody', () => {
  test('drops comment lines and everything below the scissors line', () => {
    const msg = `feat: x\n# ${FAKE_EMAIL}\n# ------------------------ >8 ------------------------\n${FAKE_EMAIL}`;

    expect(kinds(commitMessageBody(msg))).toEqual([]);
  });

  test('keeps the real message', () => {
    expect(kinds(commitMessageBody(`fix: y\n\nreported by ${FAKE_EMAIL}`))).toEqual(['email']);
  });
});

describe('isBinary', () => {
  test('detects a NUL byte', () => {
    expect(isBinary(new Uint8Array([0x89, 0x50, 0x00, 0x47]))).toBe(true);
    expect(isBinary(new TextEncoder().encode('plain text'))).toBe(false);
  });
});

describe('formatFindings', () => {
  test('names file, line and kind but never the value', () => {
    const out = formatFindings(scanText('notes.md', `a ${FAKE_EMAIL}\nb ${FAKE_TOKENS[0]}`, []));

    expect(out).toContain('notes.md:1 email');
    expect(out).toContain('notes.md:2 token');
    expect(out).not.toContain(FAKE_EMAIL);
    expect(out).not.toContain(FAKE_TOKENS[0]);
  });
});

describe('CLI --staged', () => {
  const dir = mkdtempSync(join(tmpdir(), 'privacy-scan-'));
  const script = join(import.meta.dirname, 'privacy-scan.ts');
  const run = (...args: string[]) => spawnSync('git', args, { cwd: dir });
  const scan = () => spawnSync('bun', [script, '--staged'], { cwd: dir, encoding: 'utf8' });

  run('init', '-q');
  writeFileSync(join(dir, '.privacy-denylist'), 'fictional widgets\n');

  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test('passes a clean staged file', () => {
    writeFileSync(join(dir, 'clean.md'), 'see jane@example.com\n');
    run('add', 'clean.md');

    expect(scan().status).toBe(0);
  });

  test('blocks a staged secret and prints file and kind only', () => {
    writeFileSync(join(dir, 'leak.md'), `line one\nowner ${FAKE_EMAIL}\nFictional Widgets\n`);
    run('add', 'leak.md');
    const result = scan();

    expect(result.status).toBe(1);
    expect(result.stderr).toContain('leak.md:2 email');
    expect(result.stderr).toContain('leak.md:3 denylist');
    expect(result.stderr).not.toContain(FAKE_EMAIL);
  });

  test('skips staged binary files', () => {
    run('rm', '-q', '--cached', 'leak.md');
    writeFileSync(join(dir, 'blob.bin'), Buffer.concat([Buffer.from([0]), Buffer.from(FAKE_EMAIL)]));
    run('add', 'blob.bin');

    expect(scan().status).toBe(0);
  });
});
