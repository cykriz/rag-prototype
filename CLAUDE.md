# RAG Prototype

Local, offline RAG over the Bun documentation. Full design, milestones and schema: `docs/PLAN.md`.

## Stack

- TypeScript (strict, `noUncheckedIndexedAccess`) on **Bun**
- PostgreSQL 17 + pgvector (Docker Compose) — retrieval in SQL
- Ollama (OpenAI-compatible endpoint) behind an own provider layer
- Vitest for unit/integration tests, Biome for linting

## Commands

- Install: `bun install`
- Lint: `bun run lint`
- Typecheck: `bun run typecheck`
- Database: `docker compose up -d`, then `bun run db:migrate && bun run db:embeddings` (idempotent);
  `DATABASE_URL` as in `.env.example`
- Ingest: `./scripts/fetch-corpus.sh`, then `bun run ingest [--chunk mdfm-1024-128] [--embed <model>] [--force] [--prune]`
  (idempotent; needs `CORPUS_SHA` in `.env`). Chunk keys: `md-<size>-<overlap>`, or `mdfm-<size>-<overlap>`, which
  also indexes the frontmatter `description` in each page's first chunk (M19b)
- Retrieve: `bun run retrieve "<question>" [--k 5] [--chunk mdfm-1024-128] [--embed <model>] [--search vector|fts|hybrid] [--rewrite | --expand] [--rerank]`
  (top k chunks with scores, higher is better; needs an ingest of that config and model)
- Ask: `bun run ask "<question>" [--k 5] [--chunk mdfm-1024-128] [--embed <model>] [--search vector|fts|hybrid] [--rewrite | --expand] [--rerank]` (answer with chunk citations
  or `NOT_FOUND`; writes `rag_requests` and `llm_calls` rows; needs the ingest and Ollama). `--pipeline agent` runs the
  LangGraph agent instead (≤ 4 `search_docs` calls, trace printed first; no `--rewrite`/`--expand`/`--rerank`, `--k` ≤ 10).
  `--style complete|concise` (M19c): default `complete`, the answer prompt that combines every relevant chunk; the agent
  defaults to `concise` and refuses `complete`
- Evals: `bun run eval:retrieval --config experiments/baseline.json [--split dev]` and
  `bun run eval:answers --config experiments/baseline.json [--split dev] [--seed <n>]` (answers + LLM judge; needs the
  ingest and Ollama with chat and judge model); both write `results/<timestamp>_<experiment>_<type>.json`
- Experiment matrix: `bun run experiment experiments/<file>.json [--split dev] [--seed <n>]` (every `vary` value ×
  `evals`, in process; pre-flight checks every ingest and model first; writes
  `results/<timestamp>_<experiment>_<variant>_<type>.json`). `pipeline: agent` runs the agent in the answers eval
  (answers only; no rewrite/expand/rerank, `k` ≤ 10)
- README results: `bun run report` (tables between `<!-- results:start -->`/`<!-- results:end -->` from test-split,
  clean result files, incl. the phase 2 delta table) and `bun run report --check` (exit 1 when README differs; also runs in CI)
- Validate golden dataset: `bun run eval:validate [--strict]` (schema, spans against the fetched corpus, §6 counts;
  `--strict` makes count deviations errors)
- Test: `bun run test` — Vitest projects `unit` and `integration` (`bun --bun vitest run`) plus the Claude hook
  tests (`bun test ./.claude/hooks`, they use `bun:test` and are excluded from Vitest). Integration needs the
  database and `DATABASE_URL`; it runs against `<db>_test`. DB-free: `bun run test:unit`
- Install git hooks (once per clone): `git config core.hooksPath scripts/git-hooks`
- Run lint + typecheck + test after every code change; fix all errors before moving on.

## Hard Rules

- **Bun only** — `bun`/`bunx` for install, scripts and one-off runs. No npm/yarn/pnpm.
- **Never commit or push.** Leave changes in the working tree.
- **Exact dependency versions.** List any new package (name + version) and wait for approval before installing.
- **Own provider interface** — every model call goes through `ChatProvider`/`EmbeddingProvider`; no framework
  abstraction as the call path.
- **Golden dataset references evidence spans, not chunk IDs.**
- **Evals run locally; CI uses fixtures and mocks only** (fake provider, no Ollama, no secrets).

## Plan First

Non-trivial changes start in plan mode. `ExitPlanMode` is gated by `.claude/hooks/plan-review-gate.ts`
(registered in `.claude/settings.json`): it denies the call until the `plan-reviewer` subagent has reviewed the
plan and returns `.claude/hooks/plan-review-protocol.md` as the reason. A deny is expected behaviour — follow the
protocol, then call `ExitPlanMode` again. The gate never returns `allow` (the user's approval dialog always
happens) and fails open on errors. Review criteria live only in `.claude/plan-review-criteria.md`.

Debug: `PLAN_REVIEW_GATE_DEBUG=/path/dump.jsonl` logs each hook payload plus the decision.

## Code Review

Run `/review` before every commit (final look at the finished state — fixes applied by a review are not
re-reviewed). Additionally `.claude/hooks/code-review-gate.ts`, registered in `.claude/settings.json` for
`UserPromptSubmit` (snapshot of the dirty tree) and `Stop` (check), makes the main agent run it once per user
turn that changed non-doc files; it fails open. Any review started in a turn satisfies the gate, so edits made
after a mid-turn `/review` are only caught by the pre-commit run. The procedure (blind subagent, verifying and
applying findings, running the gates) lives only in `.claude/commands/review.md`; checks live only in
`.claude/review/checks.md`.

Debug: `CODE_REVIEW_GATE_DEBUG=/path/dump.jsonl` logs each hook payload plus the decision.

## Privacy & Genericity

- This is a public-ready, generic project: **no company or personal data** in code, fixtures, docs, results or
  commit messages — no real names, emails, internal hostnames, absolute home paths, IPs or tokens.
- Test data is synthetic; use documentation values (`example.com`, `192.0.2.x`, `localhost`).
- The git hooks run `scripts/privacy-scan.ts` (emails, home paths, IPs, tokens, denylist). They print file and
  kind of finding, never the value, and **fail closed**.
- Personal/company terms go into the local, gitignored `.privacy-denylist` — case-insensitive, `*`/`?` wildcards (template:
  `.privacy-denylist.example`).
- **gitleaks is deliberately not used**; `scripts/privacy-scan.ts` alone carries secret and token detection.
