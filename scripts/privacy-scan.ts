#!/usr/bin/env bun
/**
 * Privacy gate — run by the git hooks in `scripts/git-hooks/` (pre-commit: `--staged`, commit-msg:
 * `--commit-msg <file>`). It is the repo's only secret/PII scanner: gitleaks is deliberately not used.
 *
 * Rules that are not negotiable:
 *
 * 1. **Never print the value.** Output is `file:line kind` only — a finding echoed to the terminal (or a CI
 *    log) is already a leak.
 * 2. **Fail closed.** Any error exits non-zero, so the hook blocks the commit. Only the Claude plan review
 *    hook fails open; this one must not.
 *
 * Documentation values (RFC 2606 domains, RFC 5737 IPs, loopback) are allowlisted, so synthetic test data
 * can use them literally.
 */
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

export type FindingKind = 'email' | 'home-path' | 'ipv4' | 'token' | 'denylist';

export interface Finding {
  file: string;
  line: number;
  kind: FindingKind;
}

const DENYLIST_FILE = '.privacy-denylist';
const COMMIT_MSG_LABEL = 'commit message';
const SCISSORS = '# ------------------------ >8 ------------------------';
const BINARY_SNIFF_BYTES = 8000;

const EMAIL = /[A-Za-z0-9._%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})/g;
const ALLOWED_EMAIL_DOMAIN = /(?:^|\.)(?:example\.(?:com|org|net)|example|test|invalid|localhost)$/i;
const ALLOWED_EMAIL_LOCAL = new Set(['git']);

const HOME_PATH = /(?<![\w.])(?:\/Users\/|\/home\/|[A-Za-z]:\\Users\\)[A-Za-z0-9._-]+/g;

const IPV4 = /(?<![\d.])(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})(?!\d|\.\d)/g;

const TOKENS: RegExp[] = [
  /\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}/,
  /\bgh[pousr]_[A-Za-z0-9]{36,}/,
  /\bgithub_pat_[A-Za-z0-9_]{22,}/,
  /\bxox[abprs]-[A-Za-z0-9-]{10,}/,
  /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/,
  /\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/,
  /\bBearer\s+[A-Za-z0-9._~+/-]{20,}/,
  /-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----/,
  /\b(?:api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['"][^'"\s]{12,}['"]/i,
];

function isAllowedEmail(match: string, domain: string): boolean {
  const local = match.slice(0, match.indexOf('@'));

  return ALLOWED_EMAIL_DOMAIN.test(domain) || ALLOWED_EMAIL_LOCAL.has(local.toLowerCase());
}

function isAllowedIpv4(octets: number[]): boolean {
  const [a, b, c] = octets;
  const prefix = `${a}.${b}.${c}`;

  if (a === 127) {
    return true;
  }

  if (octets.every((o) => o === 0)) {
    return true;
  }

  return prefix === '192.0.2' || prefix === '198.51.100' || prefix === '203.0.113';
}

function hasEmail(line: string): boolean {
  for (const m of line.matchAll(EMAIL)) {
    if (!isAllowedEmail(m[0], m[1] ?? '')) {
      return true;
    }
  }

  return false;
}

function hasIpv4(line: string): boolean {
  for (const m of line.matchAll(IPV4)) {
    const octets = m.slice(1, 5).map(Number);

    if (octets.some((o) => o > 255)) {
      continue;
    }

    if (!isAllowedIpv4(octets)) {
      return true;
    }
  }

  return false;
}

/** `*` = any run of characters, `?` = exactly one; everything else is literal. */
function entryToPattern(entry: string): RegExp {
  const source = entry
    .split('')
    .map((ch) => {
      if (ch === '*') return '.*';
      if (ch === '?') return '.';
      return ch.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    })
    .join('');

  return new RegExp(source, 'i');
}

/**
 * Parses `.privacy-denylist`: one case-insensitive pattern per line, matched anywhere in a line, with `*` and
 * `?` as wildcards. `#` comments and blanks are ignored, and so are entries made of wildcards only — they
 * would block every line.
 */
export function parseDenylist(text: string): RegExp[] {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith('#') && /[^*?]/.test(l))
    .map(entryToPattern);
}

/** At most one finding per line and kind — the count matters, the duplicates don't. */
export function scanText(file: string, text: string, denylist: RegExp[]): Finding[] {
  const findings: Finding[] = [];

  for (const [index, line] of text.split('\n').entries()) {
    const at = (kind: FindingKind) => findings.push({ file, line: index + 1, kind });
    if (hasEmail(line)) at('email');
    if (line.match(HOME_PATH) !== null) at('home-path');
    if (hasIpv4(line)) at('ipv4');
    if (TOKENS.some((re) => re.test(line))) at('token');
    if (denylist.some((pattern) => pattern.test(line))) at('denylist');
  }

  return findings;
}

/** Git strips `#` lines and everything below the scissors line; scan only what will be committed. */
export function commitMessageBody(text: string): string {
  const cut = text.indexOf(SCISSORS);
  const kept = cut < 0 ? text : text.slice(0, cut);

  return kept
    .split('\n')
    .map((l) => (l.startsWith('#') ? '' : l))
    .join('\n');
}

/** Same heuristic as git: a NUL byte in the first 8000 bytes means binary. */
export function isBinary(bytes: Uint8Array): boolean {
  return bytes.subarray(0, BINARY_SNIFF_BYTES).includes(0);
}

export function formatFindings(findings: Finding[]): string {
  const lines = findings.map((f) => `privacy-scan: ${f.file}:${f.line} ${f.kind}`);

  lines.push(
    `privacy-scan: ${findings.length} finding(s) — values are not printed. Remove them or replace them with ` +
      'synthetic data (example.com, 192.0.2.x); denylist entries live in .privacy-denylist.',
  );

  return lines.join('\n');
}

/** Runs git with the given args (callers pass `-C <dir>` when needed); throws on a non-zero exit. */
export function git(args: string[]): Buffer {
  const result = spawnSync('git', args, { maxBuffer: 256 * 1024 * 1024 });

  if (result.status !== 0) {
    throw new Error(`git ${args[0]} failed (exit ${result.status})`);
  }

  return result.stdout;
}

function loadDenylist(root: string): RegExp[] {
  const path = join(root, DENYLIST_FILE);

  if (!existsSync(path)) {
    console.error(`privacy-scan: no ${DENYLIST_FILE} — denylist check skipped (see ${DENYLIST_FILE}.example)`);
    return [];
  }

  return parseDenylist(readFileSync(path, 'utf8'));
}

function scanStaged(denylist: RegExp[]): Finding[] {
  const names = git(['diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'])
    .toString('utf8')
    .split('\0')
    .filter((n) => n.length > 0);
  const findings: Finding[] = [];

  for (const name of names) {
    const blob = git(['show', `:${name}`]);

    if (isBinary(blob)) {
      continue;
    }

    findings.push(...scanText(name, blob.toString('utf8'), denylist));
  }

  return findings;
}

function main(argv: string[]): number {
  const root = git(['rev-parse', '--show-toplevel']).toString('utf8').trim();
  const denylist = loadDenylist(root);
  const [mode, arg] = argv;
  let findings: Finding[];

  if (mode === '--staged') {
    findings = scanStaged(denylist);
  } else if (mode === '--commit-msg' && arg !== undefined) {
    findings = scanText(COMMIT_MSG_LABEL, commitMessageBody(readFileSync(arg, 'utf8')), denylist);
  } else {
    console.error('usage: privacy-scan.ts --staged | --commit-msg <file>');
    return 2;
  }

  if (findings.length === 0) {
    return 0;
  }

  console.error(formatFindings(findings));
  return 1;
}

if (import.meta.main) {
  try {
    process.exit(main(process.argv.slice(2)));
  } catch (error) {
    // Fail closed. Print only the message: never file contents.
    console.error(`privacy-scan: error — commit blocked: ${error instanceof Error ? error.message : 'unknown'}`);
    process.exit(2);
  }
}
