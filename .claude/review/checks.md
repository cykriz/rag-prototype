# Code Review Checks

The single source of code review rules for this repo. Consumer: the subagent `review-changes` (started by
`/review`), which reports **deviations only** in the format from `.claude/agents/review-changes.md`.

Project invariants are **not** repeated here: apply `.claude/plan-review-criteria.md` § "Project Invariants"
to the diff as well.

## Correctness & Bugs

- Logic errors, off-by-one, wrong conditions, unreachable or dead branches.
- Unhandled error and edge paths (empty input, missing file, model unreachable, malformed model output).
- Broken contracts with callers: changed signatures, return shapes or exit codes without updating every caller.
- Tests that do not exercise the changed behaviour, or assert on something other than what changed.

## TypeScript/Bun specifics

- Strict mode holds: no `any`, `@ts-ignore`/`@ts-expect-error` or non-null `!` used to silence the compiler.
- No Bun-only APIs in code Vitest runs on Node: `bun:test` imports only under `.claude/hooks` (excluded in
  `vitest.config.ts`). `bun run test` uses `bun --bun vitest`, so Vitest itself runs on Bun — flag `Bun.*`
  only where a Node execution path actually exists, not as a blanket ban.
- `noUncheckedIndexedAccess`: indexed access is handled by narrowing (guard, `?.`, default with a reason), not by
  assertion.

## Duplication & Unnecessary Complexity

- Every new artifact (file, export, script, config key) must name what breaks if it is deleted. If the answer
  is "nothing", it is a finding.
- Single-caller abstractions: a new helper, wrapper or interface with exactly one caller — grep for others.
- Silent fallbacks: swallowed errors, defaults that mask a failure, `catch` without rethrow or logging.
- Copied rules: a rule, constant or logic block written out again instead of referencing its source. Duplication
  findings name both sides (`a.ts:10-30 vs b.ts:40-60`).

## Privacy & Genericity

- Scope: `.claude/plan-review-criteria.md` § "Project Invariants" → "Privacy & genericity", applied to code,
  docs, tests and fixtures in the diff.
- Findings name `file:line` and the kind (email, home path, IP, token, name, company term) and **never quote the
  value**.

## Documentation Sync

- `CLAUDE.md`, README and `docs/PLAN.md` must stay true when a change adds or alters commands, env vars or
  milestones.
- Only extend a doc where a statement would otherwise be wrong or a new command/env var would be undocumented —
  no general doc expansion.

## Severity Floors

Minimums, not ceilings — raise them when the damage warrants it.

- Duplicated rule or logic → **high**
- Personal or company data → **high**
- New artifact without named breakage → **high**
- Doc made false by the change → **high**
- New command or env var without doc → **medium**
