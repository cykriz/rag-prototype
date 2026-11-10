# Plan Review Criteria

The single source of plan review rules for this repo. Consumer: the subagent `plan-reviewer` (automatic
in plan mode before `ExitPlanMode`), which reports **deviations only**, in the compact findings format
from `.claude/agents/plan-reviewer.md`.

Three parts are always reported, never deviations-only: **baseline**, **deletion inventory** and
**simplicity line** (see "Duplication & Simplicity").

## How to run this review

1. **`CLAUDE.md` is already in your context** — judge against it and do **not** read it again. If it is
   missing there, report that as `Gap:`.
2. **Read this file in full.** Not from memory.
3. **`docs/PLAN.md` is the reference** for milestones, directory layout, schema and pinned versions. Read
   only the sections the plan touches.
4. **Evidence, not assertions**: back every claim about the existing code or docs with `Read`/`Grep` and
   cite it as `file:line`. Mark what you did not verify.
5. **No Bash.** You cannot run tests, queries or models; judge what the plan says it will run.

## Generic Quality

- Completeness & scope — solves the original request, no more, no less
- Assumptions stated explicitly
- Step order / dependencies clear (e.g. migration before code that reads the table)
- Error and edge cases considered (empty corpus, model unreachable, malformed model output, NOT_FOUND)
- Verification — the plan names the commands it needs (`bun run lint`, `bun run typecheck`,
  `bun run test`, plus milestone-specific checks) and new logic gets unit tests
- Rollback / reversibility (migrations idempotent, re-runs produce no change)
- Affected files named concretely

## Project Invariants

The rules live in `CLAUDE.md` (in your context); these lines are the checklist, not a second copy. Mark
invariants that don't apply as `n/a` instead of omitting them.

- **Privacy & genericity** — no company or personal data enters the repo: no real names, emails,
  internal hostnames, absolute home paths, IPs, tokens or customer content in code, fixtures, docs,
  commit messages or results. Test data is synthetic. Anything the privacy gate
  (`scripts/privacy-scan.ts`) would flag is a finding at plan time already.
- **Golden dataset references evidence spans, not chunk IDs** — `eval/golden.jsonl` entries point at
  source text (document + quoted span); relevance is resolved against chunks at eval time
  (`src/eval/relevance.ts`). A plan that stores chunk IDs in the dataset ties it to one chunking config
  and breaks every chunking experiment.
- **Own provider interface** — model calls go through `ChatProvider` / `EmbeddingProvider`
  (`src/llm/types.ts`) and the registry. No framework abstraction (LangChain chat models, LlamaIndex,
  Vercel AI SDK) as the call path; LangChain/LangGraph only as a library behind or beside that interface
  (text splitting, graph orchestration calling our providers).
- **Evals local, CI on fixtures and mocks** — retrieval/answer evals and anything needing Ollama or a
  real API key run locally only. CI uses the fake provider, fixtures and the pgvector service container;
  no model downloads, no secrets, no network calls to model endpoints.
- **Bun only** — `bun`/`bunx` for install, scripts and one-off runs; no npm/yarn/pnpm lockfiles.
- **Packages** — exact versions; any new package is listed for approval before install.
- **Retrieval inspectable in SQL** — vector and full text retrieval live in PostgreSQL 17/pgvector and
  stay reproducible with plain SQL; no external vector store, no in-process index replacing it.
- **Results versioned** — experiments write a versioned JSON result (config, dataset hash, corpus SHA);
  README numbers are generated from results, never typed by hand.
- **Plan first** — non-trivial changes start in plan mode; the plan stays within one milestone of
  `docs/PLAN.md` unless it says why.

## Duplication & Simplicity (Plan Level)

Run these against what the plan **intends to do**. They are actively run — grep, read both sides. A probe
you did not run must not be reported as "found nothing".

- **Deletion inventory** (runs first): for **every artifact the plan proposes** (file, export, script,
  table, config key) name what breaks if it is left out. An artifact whose answer is "nothing" is a
  finding.
- **Baseline:** name the smallest implementation that meets the *original request* — its mechanism in
  one clause, plus new files / new exports / changed files — and measure the plan against it. Every item
  above the baseline needs its own reason.
- **Prior art** — the probe that most often decides a plan's worth. For every snippet, helper or
  described behaviour in the plan, grep the repo for code that already does it (by behaviour keywords,
  not just by name). Writing out logic an existing function already performs is a finding.
- **Extraction**: wording like "split out of", "moved to", or a new module mirroring an existing file,
  with no sentence saying the original goes away. Reverse direction: a new shared helper with exactly one
  caller — grep for the other copies.
- **Special-case lists**: branches per model/provider/table that a stricter reading of the request
  collapses into one path.
- **Mode parameters**: a new boolean/mode parameter that forks a function body — the plan says how much
  differs, or proposes two functions.
- **Implicit ordering**: steps whose correctness depends on "must run before X" without types or
  migrations enforcing it.
- **Counts**: a plan's numbers about the *existing* code are claims — re-run the grep and say what you
  re-counted.
- **Cross-references in the plan's wording** ("same as X", "mirrors Y"): check X/Y actually exist and
  match.

**Always answer "is this the simplest way?"** — once, as the aggregate. It judges the **mechanism
itself**: whether the chosen approach is needed at all, anchored on a `file:line` in the existing code or
`docs/PLAN.md` rather than on the plan's own reasoning. A `no` names the concrete simpler alternative and
is **also** a finding.

## Severity

`high` / `medium` / `low`. Minimums, not ceilings — raise them when the damage warrants it.

- The plan breaks a project invariant, or is not implementable as written → **high**
- Company or personal data could enter the repo → **high**
- The plan writes out logic an existing function already performs → **high**
- The simplicity line answers *no* → **high**
- A gap that would surface in code review at the latest (missing verification, open edge case,
  unlisted package) → **medium**
- Naming, ordering of prose, minor unclear wording → **low**
- **No `approve` while any inventory item has no named breakage** — even when everything else is clean

## Not Checkable at the Plan Level

These need finished code and remain the job of code review. **No** findings are produced for them —
guessing looks plausible and is wrong anyway:

nesting depth, number of decision points, live variables held at once, reading span, comment load, naming
of individual symbols, actual retrieval or answer quality numbers.
