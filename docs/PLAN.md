# RAG Prototype: Project Plan

Status: plan only. All package versions were looked up on 2026-11-10 (`npm view`) and are re-verified under Bun in M0.

Declared assumptions (apply throughout):
- Assumption: development machine is an Apple Silicon MacBook Pro with 32 GB unified memory, Ollama installed natively via Homebrew.
- Assumption: Bun docs in `oven-sh/bun` are covered by the repository's MIT license (`LICENSE.md`). M6 verifies this and records the result in the README.
- Assumption: `openai@7.x` still exposes `chat.completions.create`, `embeddings.create` and the `openai/helpers/zod` helpers. M0 and M1 verify this before any adapter code is written. Verified in M1, see `docs/SPIKE.md`.

---

## 1. Target picture and architecture

A local, offline RAG system answers questions about the Bun documentation, cites the chunk IDs it used and replies `NOT_FOUND` when the context is insufficient. Every model call goes through a thin own provider layer (OpenAI compatible adapter pointed at Ollama by default, Anthropic adapter optional) with retries, token counting, latency and cost logging. Retrieval runs entirely in PostgreSQL 17 with pgvector HNSW indexes and full text search, so retrieval quality can be inspected with plain SQL. A golden dataset written by hand drives retrieval metrics (Recall@k, MRR, nDCG, keyword coverage) and LLM-as-a-judge answer metrics, and every experiment writes a versioned JSON result that generates the README table. Phase 2 adds hybrid search, query rewriting, query expansion, LLM reranking and a LangGraph agent, each measured against the phase 1 baseline, including techniques that did not help.

```
                         ┌──────────────────────── Bun CLI (src/cli) ───────────────────────┐
                         │  ingest   ask   eval:retrieval   eval:answers   experiment  report │
                         └───────┬──────────┬───────────────────┬──────────────┬─────────────┘
                                 │          │                   │              │
   corpus/bun-docs (pinned SHA)  │          ▼                   ▼              ▼
          │                      │   ┌─────────────┐     ┌────────────┐  results/*.json ──► README table
          ▼                      │   │ RAG pipeline│     │ eval module│
  ┌──────────────┐               │   │ rewrite?    │     │ metrics.ts │
  │ ingest       │               │   │ expand?     │     │ judge.ts   │
  │ load md      │               │   │ retrieve    │◄────┤ golden.jsonl
  │ split (LC.js)│               │   │ rerank?     │     └─────┬──────┘
  │ hash, embed  │               │   │ answer      │           │
  └──────┬───────┘               │   └──┬───────┬──┘           │
         │                       │      │       │  (phase 2: LangGraph agent wraps retrieve as a tool)
         │        ┌──────────────┴──────┴───┐   │
         │        │ provider layer (src/llm) │◄──┴────────────┘
         │        │ ChatProvider             │  retry + backoff, Zod structured output,
         │        │ EmbeddingProvider        │  token count, latency, cost ──► llm_calls
         │        └──────┬────────────┬──────┘
         │               │            │ (phase 3, optional)
         │               ▼            ▼
         │      Ollama :11434/v1   Anthropic API
         │      (openai npm pkg)   (@anthropic-ai/sdk)
         ▼
  ┌───────────────────────────────────────────────┐
  │ PostgreSQL 17 + pgvector (Docker Compose)      │
  │ documents, chunks(tsv GIN), emb_<model>(HNSW), │
  │ llm_calls, rag_requests                        │
  └───────────────────────────────────────────────┘
```

---

## 2. Covered topics

| Topic | Project part | Where in the repo |
|---|---|---|
| LLM APIs in production: token limits, latency, cost | Provider layer, call logging | `src/llm/retry.ts`, `src/llm/tokens.ts`, `src/obs/logger.ts`, `llm_calls` table, README KPI block (p50/p95, tokens, cost per 100 questions) |
| TypeScript backend | Entire codebase | `tsconfig.json` with `strict`, `noUncheckedIndexedAccess`; CI typecheck job |
| SQL | Schema, vector and hybrid queries, KPI queries | `migrations/001_init.sql`, `src/retrieval/*.ts`, `src/obs/kpis.sql` |
| Docker | Local environment, CI service container | `docker-compose.yml`, `.github/workflows/ci.yml` |
| CI/CD | Lint, typecheck, unit and integration tests | `.github/workflows/ci.yml`, green badge in README |
| Cloud fundamentals | Phase 4 only | Covered only once phase 4 is live |
| AI generated tests | Tests written with Claude Code, reviewed by me | `tests/`, README section "How this repo was built" describing the review process |
| Evaluating AI output quality | Golden dataset, retrieval metrics, judge | `eval/golden.jsonl`, `src/eval/metrics.ts`, `src/eval/judge.ts`, `results/` |
| Agentic concepts, RAG pipelines, LangGraph | Phase 2 agent | `src/agent/graph.ts`, `src/agent/tools.ts`, agent row in results table |
| KPIs for AI workflow impact | Observability and results table | README "Results" and "KPIs" sections generated from `results/` |
| Chunking strategies | Chunker plus chunk size experiment | `src/ingest/chunk.ts`, experiment `exp-chunk-size` |
| Embeddings, vector stores | pgvector, two embedding models | `emb_nomic_embed_text`, `emb_mxbai_embed_large`, experiment `exp-embed-model` |
| Query rewriting, query expansion | Phase 2 | `src/rag/rewrite.ts`, `src/rag/expand.ts` |
| Reranking | Phase 2 LLM reranker with structured output | `src/rag/rerank.ts` |
| MRR, nDCG, golden datasets | Eval module | `src/eval/metrics.ts` with unit tests on hand computed examples |
| LLM-as-a-judge | Answer eval | `src/eval/judge.ts`, judge prompt in `src/eval/prompts.ts` |
| Structured outputs | Zod everywhere a model returns data | `src/llm/structured.ts`, `docs/SPIKE.md` |
| Tool calling, agent loop | LangGraph agent with `search_docs` tool | `src/agent/` |
| OpenAI compatible endpoints with Ollama | Default adapter | `src/llm/openaiCompatible.ts` with `baseURL=http://localhost:11434/v1` |
| Cost and token management | Pricing config and cost estimate | `config/pricing.json`, `src/llm/pricing.ts` |

---

## 3. Repo structure

```
rag-prototype/
├── README.md                     Overview, setup, results table, KPIs, limitations and learnings
├── LICENSE                       MIT for own code
├── CLAUDE.md                     Project rules for the coding agent (baseline)
├── package.json                  Scripts (bun run ...), exact dependency versions (baseline)
├── bun.lock                      Lockfile (Bun only, no package-lock.json) (baseline)
├── tsconfig.json                 strict, noUncheckedIndexedAccess, moduleResolution bundler (baseline)
├── biome.json                    Lint and format config (baseline)
├── vitest.config.ts              Baseline excludes .claude/**; unit and integration projects added in M4
├── .privacy-denylist.example     Template for the local, gitignored privacy denylist (baseline)
├── .claude/                      Plan review gate, plan-reviewer agent, review criteria (baseline)
├── docker-compose.yml            Postgres (pgvector/pgvector:pg17); optional ollama profile for Linux
├── .env.example                  All variables with defaults, no secrets
├── config/
│   ├── models.json               Model tags and embedding dims per profile (default, light)
│   └── pricing.json              Price per 1M input/output tokens per model (local models = 0)
├── migrations/
│   ├── 001_init.sql              Extension, documents, chunk_configs, chunks, llm_calls, rag_requests
│   └── templates/embeddings.sql.tpl   Per model embeddings table with fixed vector(N) and HNSW index
├── scripts/
│   ├── fetch-corpus.sh           Sparse clone of oven-sh/bun docs at pinned SHA into corpus/ (gitignored)
│   ├── spike-ollama.ts           M1: checks structured output paths, tools, embeddings, usage fields
│   ├── privacy-scan.ts           Secret and PII gate run by the git hooks (baseline)
│   └── git-hooks/                pre-commit and commit-msg hooks (baseline)
├── src/
│   ├── config.ts                 Env parsing with Zod, single source of configuration
│   ├── llm/
│   │   ├── types.ts              ChatProvider, EmbeddingProvider, message and tool types
│   │   ├── openaiCompatible.ts   Adapter 1: openai npm package with baseURL (Ollama)
│   │   ├── anthropic.ts          Adapter 2 (phase 3): @anthropic-ai/sdk, chat only
│   │   ├── fake.ts               Deterministic fake providers for tests
│   │   ├── registry.ts           Picks adapters from CHAT_PROVIDER and EMBED_PROVIDER
│   │   ├── structured.ts         generateObject(schema): json_schema, fallback json_object, Zod, retry
│   │   ├── retry.ts              Exponential backoff with full jitter, retryable error classification
│   │   ├── tokens.ts             Usage fields first, js-tiktoken estimate as fallback
│   │   └── pricing.ts            Cost estimate from pricing.json
│   ├── obs/
│   │   ├── logger.ts             Writes llm_calls and rag_requests rows
│   │   └── kpis.sql              p50/p95 latency, tokens, cost per 100 questions
│   ├── db/
│   │   ├── client.ts             postgres package connection
│   │   ├── migrate.ts            Applies migrations once, records them in schema_migrations
│   │   └── embeddingTables.ts    Renders the template per model, validates dim against config
│   ├── ingest/
│   │   ├── loadMarkdown.ts       Reads .md/.mdx, strips frontmatter and JSX components, extracts title
│   │   ├── chunk.ts              RecursiveCharacterTextSplitter (markdown) with token length function
│   │   ├── hash.ts               sha256 via node:crypto
│   │   └── ingest.ts             Idempotent upsert of documents, chunks, embeddings
│   ├── retrieval/
│   │   ├── vector.ts             Top k by cosine distance
│   │   ├── fts.ts                Top k via websearch_to_tsquery (phase 2)
│   │   ├── hybrid.ts             Reciprocal rank fusion in SQL (phase 2)
│   │   └── pgRetriever.ts        LangChain BaseRetriever wrapper around the above
│   ├── rag/
│   │   ├── prompts.ts            Answer prompt, NOT_FOUND rule, citation format
│   │   ├── schemas.ts            Zod schemas: Answer, Rewrite, Expansion, Rerank
│   │   ├── answer.ts             Retrieve, build context, generate, validate citations
│   │   ├── rewrite.ts            Phase 2
│   │   ├── expand.ts             Phase 2
│   │   └── rerank.ts             Phase 2
│   ├── agent/
│   │   ├── graph.ts              Phase 2: LangGraph StateGraph (agent node, tools node, stop rule)
│   │   └── tools.ts              search_docs tool definition and executor
│   ├── eval/
│   │   ├── dataset.ts            Zod schema for golden.jsonl, loader, dataset hash
│   │   ├── relevance.ts          Evidence span matching against chunks
│   │   ├── metrics.ts            Recall@k, MRR, nDCG@k, keyword coverage, abstention metrics
│   │   ├── judge.ts              LLM-as-a-judge with structured output
│   │   ├── prompts.ts            Judge prompt
│   │   └── runner.ts             Runs a config against the dataset, writes results JSON
│   └── cli/
│       ├── ingest.ts             bun run ingest --chunk md-512-64 --embed nomic-embed-text
│       ├── ask.ts                bun run ask "question"
│       ├── eval.ts               bun run eval:retrieval / eval:answers --config experiments/x.json
│       ├── experiment.ts         Runs an experiment matrix file
│       └── report.ts             Regenerates README results table between markers
├── eval/
│   └── golden.jsonl              45 hand written questions (dev and test split)
├── experiments/                  Experiment definitions (JSON), one file per experiment
├── results/                      One JSON per run (committed), source of README numbers
├── tests/
│   ├── unit/                     deps (Bun import check), chunk, hash, metrics, relevance, retry, structured, rrf, pricing
│   ├── integration/              ingest + retrieval + answer against real Postgres, fake models
│   └── fixtures/
│       ├── docs/                 5 to 8 small hand written Markdown files
│       └── llm/                  Recorded model responses for mocked calls
├── docs/
│   ├── PLAN.md                   This plan
│   ├── SPIKE.md                  M0 model check (RAM, speed) and M1 findings on Ollama capabilities
│   └── adr/0001 ... 0006         Architecture decision records
└── .github/workflows/ci.yml      Typecheck, lint, unit, integration with pgvector service
```

`.gitignore` already contains `notes/`, `node_modules/`, `.env`, `.privacy-denylist`; M6 adds `corpus/`.

---

## 4. Database schema

Design: chunks are independent of the embedding model. Each embedding model gets its own table with a fixed `vector(N)` column, so the HNSW index always matches the column type and dimension mismatches fail loudly at insert time.

`migrations/001_init.sql`:

```sql
BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;          -- pgvector >= 0.8 required (iterative index scans)

CREATE TABLE IF NOT EXISTS schema_migrations (
  version     text PRIMARY KEY,
  applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_path     text NOT NULL UNIQUE,                 -- relative to corpus root
  title           text,
  content_hash    char(64) NOT NULL,                    -- sha256 of normalized file content
  corpus_version  text NOT NULL,                        -- pinned git SHA of oven-sh/bun
  ingested_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunk_configs (
  key            text PRIMARY KEY,                      -- e.g. 'md-512-64'
  chunk_size     int  NOT NULL CHECK (chunk_size BETWEEN 64 AND 4096),
  chunk_overlap  int  NOT NULL CHECK (chunk_overlap >= 0),
  splitter       text NOT NULL,
  CHECK (chunk_overlap < chunk_size)
);

CREATE TABLE IF NOT EXISTS chunks (
  id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  document_id   bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_config  text   NOT NULL REFERENCES chunk_configs(key),
  ordinal       int    NOT NULL CHECK (ordinal >= 0),
  heading_path  text   NOT NULL DEFAULT '',             -- "Runtime > Environment variables"
  content       text   NOT NULL CHECK (length(content) > 0),
  content_hash  char(64) NOT NULL,
  token_count   int    NOT NULL CHECK (token_count > 0),
  tsv           tsvector GENERATED ALWAYS AS
                  (to_tsvector('english', heading_path || ' ' || content)) STORED,
  UNIQUE (document_id, chunk_config, ordinal)
);
CREATE INDEX IF NOT EXISTS chunks_config_idx ON chunks (chunk_config);
CREATE INDEX IF NOT EXISTS chunks_hash_idx   ON chunks (content_hash);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx    ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS rag_requests (
  request_id        uuid PRIMARY KEY,
  run_id            text NOT NULL,
  question_id       text,                               -- null for ad hoc questions
  pipeline          text NOT NULL,                      -- e.g. 'baseline', 'hybrid+rerank', 'agent'
  total_latency_ms  int  NOT NULL CHECK (total_latency_ms >= 0),
  retrieved_ids     bigint[] NOT NULL DEFAULT '{}',
  found             boolean,
  created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS llm_calls (
  id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  request_id     uuid,
  run_id         text NOT NULL,
  operation      text NOT NULL CHECK (operation IN
                   ('embed','generate','judge','rewrite','expand','rerank','agent_step')),
  provider       text NOT NULL,
  model          text NOT NULL,
  input_tokens   int CHECK (input_tokens >= 0),
  output_tokens  int CHECK (output_tokens >= 0),
  token_source   text NOT NULL CHECK (token_source IN ('usage','tiktoken_estimate')),
  latency_ms     int  NOT NULL CHECK (latency_ms >= 0),
  cost_usd       numeric(12,6) NOT NULL DEFAULT 0,
  attempts       int  NOT NULL DEFAULT 1 CHECK (attempts >= 1),
  status         text NOT NULL CHECK (status IN ('ok','error','invalid_output')),
  created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS llm_calls_run_idx ON llm_calls (run_id, operation);

INSERT INTO schema_migrations (version) VALUES ('001_init') ON CONFLICT DO NOTHING;

COMMIT;
```

`migrations/templates/embeddings.sql.tpl` (rendered by `bun run db:embeddings`, `{{table}}` validated against `^emb_[a-z0-9_]+$`, `{{dim}}` from `config/models.json`):

```sql
CREATE TABLE IF NOT EXISTS {{table}} (
  chunk_id      bigint PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  chunk_config  text   NOT NULL,                 -- denormalized for filtered ANN search
  content_hash  char(64) NOT NULL,               -- allows reuse of vectors for unchanged chunks
  embedding     vector({{dim}}) NOT NULL
);
CREATE INDEX IF NOT EXISTS {{table}}_hnsw ON {{table}}
  USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS {{table}}_cfg ON {{table}} (chunk_config);
```

Rendered instances: `emb_nomic_embed_text` with `vector(768)`, `emb_mxbai_embed_large` with `vector(1024)`.

Vector query (`src/retrieval/vector.ts`):

```sql
BEGIN;
SET LOCAL hnsw.ef_search = 100;
SET LOCAL hnsw.iterative_scan = relaxed_order;   -- keeps k results despite the chunk_config filter
SELECT * FROM (
  SELECT c.id, c.content, c.heading_path, d.source_path,
         e.embedding <=> $1::vector AS distance
  FROM emb_nomic_embed_text e
  JOIN chunks c    ON c.id = e.chunk_id
  JOIN documents d ON d.id = c.document_id
  WHERE e.chunk_config = $2
  ORDER BY e.embedding <=> $1::vector
  LIMIT $3
) t ORDER BY distance;                            -- relaxed_order can return slightly unordered rows
COMMIT;
```

Consequences of the configurable dimension (documented in README and ADR 0005):
- Changing the embedding model means a new table and a full re-embedding; vectors of different models are never comparable, so mixing them in one column would be a bug, not a feature.
- `embeddingTables.ts` checks at startup that `atttypmod` of `embedding` equals the configured dim and that the first returned embedding has that length; a mismatch aborts before any insert.
- HNSW in pgvector indexes `vector` up to 2000 dimensions; larger models would need `halfvec`. Both planned models are below that limit.
- Idempotent ingest: unchanged `documents.content_hash` means skip; changed hash means delete the document's chunks (cascade) and reinsert; embeddings for chunks whose `content_hash` already exists in the model table are copied instead of recomputed.

---

## 5. Milestones

Every milestone ends with the generic gate `bun run lint && bun run typecheck && bun run test` green; it is not repeated per milestone. Each milestone is sized for one Claude Code session and has one `Acceptance command` line (targeted test path or `bun run` script). Milestones marked "me" are done by me, not by an agent. Durations are estimates. Project setup (package.json, tsconfig, Vitest, Biome, git hooks, plan gate) is already done in the baseline commit and is not part of any milestone.

### Phase 1: MVP, fully local (about 25 h, 3 days)

**M0. Dependency and model check (1 h)**
- Tasks: list for approval, then install with exact versions: `openai@7.55.0`, `@langchain/core@1.3.6`, `@langchain/textsplitters@1.0.2`, `@langchain/langgraph@1.7.3`, `postgres@4.1.0`, `pgvector@0.3.1`, `js-tiktoken@1.0.21` (`zod`, `vitest`, `typescript`, Biome and `@types/bun` are already in the baseline; `@anthropic-ai/sdk` waits for optional M20). Write `tests/unit/deps.test.ts`: imports every planned package and exercises one offline call each (split a string, `z.toJSONSchema`, encode tokens with `js-tiktoken`, compile an empty `StateGraph`, instantiate `OpenAI` with a dummy baseURL, import `zodResponseFormat` from `openai/helpers/zod`, construct a `@langchain/core` `Document`, construct a `postgres()` client without a query, `pgvector.toSql([1,2,3])`) and asserts `typeof Bun !== 'undefined'`. Because it is a Vitest test, it also guards Bun compatibility in CI on every later dependency bump.
- Model check (precondition `brew services start ollama`): one `ollama pull <model>` each for `qwen2.5:7b`, `gemma3:12b`, `nomic-embed-text`, `mxbai-embed-large`; `ollama run --verbose qwen2.5:7b "Say ok"` and the same for `gemma3:12b` to record the eval rate; load both and record `ollama ps` sizes in a "Models" table in `docs/SPIKE.md`.
- Acceptance: deps test passes under the Bun runtime. Both chat models fit in memory together and `qwen2.5:7b` reaches at least 15 tokens/s (Assumption: threshold chosen so 45 eval answers finish well under an hour); otherwise switch to the light profile and record that in `docs/SPIKE.md`.
- Acceptance command: `bun run test` (includes `tests/unit/deps.test.ts`), plus the Models table present in `docs/SPIKE.md`.

**M1. Ollama capability spike (1 h)**
- Tasks: `scripts/spike-ollama.ts` checks against `http://localhost:11434/v1`: (a) `response_format: { type: "json_schema", json_schema: {...} }` with `qwen2.5:7b` and `gemma3:12b`; (b) Zod 4 compatibility of the openai SDK helper: `zodResponseFormat` from `openai/helpers/zod` with a Zod 4 schema and `chat.completions.parse`, on both chat models; (c) fallback: `z.toJSONSchema(schema)` passed manually as `response_format.json_schema.schema`, then own validation with `schema.safeParse`, on both chat models; (d) `tools` with one function on `qwen2.5:7b`; (e) `embeddings.create` returns 768 and 1024 floats; (f) presence of `usage` in chat and embedding responses; (g) `response_format: { type: "json_object" }` plus `safeParse`, on both chat models. Findings go to `docs/SPIKE.md`.
- Fallback rule: if (b) fails with Zod 4, use (c) with up to 2 repair retries that append the Zod error. Only if `json_schema` itself fails (a), use `json_object` (g) plus the same validation and repair retries.
- Exit code rule: exit 0 requires, for each chat model (`qwen2.5:7b` generator, `gemma3:12b` judge), (b) or (c) to pass, or else (g) to pass with `docs/SPIKE.md` stating that this model falls back to `json_object`; plus (d) on `qwen2.5:7b`, (e) and (f).
- Acceptance: `docs/SPIKE.md` has a pass or fail row for (a) to (g), the Ollama version, and the chosen structured output path per model.
- Acceptance command: `bun scripts/spike-ollama.ts` exits 0.

**M2. Config, provider interfaces, OpenAI compatible adapter, retry (2 h)**
- Tasks: `src/config.ts` (Zod env schema) and `.env.example` (all variables with defaults, no secrets); `src/llm/types.ts`:
  ```ts
  export interface ChatProvider {
    generate(req: GenerateRequest): Promise<GenerateResult>;          // messages, tools?, temperature
    generateObject<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>>;
    judge<T>(req: GenerateRequest, schema: z.ZodType<T>): Promise<ObjectResult<T>>; // judge model
  }
  export interface EmbeddingProvider {
    readonly model: string; readonly dim: number;
    embed(texts: string[], kind: "document" | "query"): Promise<EmbedResult>;
  }
  ```
  `openaiCompatible.ts` with `new OpenAI({ baseURL, apiKey: "ollama", maxRetries: 0 })` (SDK retries off so only our retry runs); `structured.ts` implementing the path chosen in M1 (`docs/SPIKE.md`, section M1); `retry.ts` (base 500 ms, factor 2, full jitter, max 4 attempts, retry on 429, 5xx, timeouts, `ECONNREFUSED`; never on 400); `registry.ts` reading `CHAT_PROVIDER`, `EMBED_PROVIDER`; `fake.ts`.
- Acceptance: `bun run ask:raw "hello"` returns text from Ollama; switching `CHAT_PROVIDER=fake` needs no code change; nomic prefixes `search_document: ` and `search_query: ` are applied inside the adapter based on `kind`. Unit tests: retry with a fake that fails twice with 503 then succeeds (3 attempts, delays increasing); no retry on 400; `generateObject` repairs invalid JSON up to 2 times and then validates or reports `invalid_output`.
- Acceptance command: `bun --bun vitest run tests/unit/retry.test.ts tests/unit/structured.test.ts && bun run ask:raw "hello"`

**M3. Token counting, latency, cost logging (1 h)**
- Tasks: `tokens.ts` (use `usage` when present, else `js-tiktoken` `cl100k_base` estimate with `token_source='tiktoken_estimate'`); `pricing.ts` with `config/pricing.json` (local models priced at 0); `obs/logger.ts` with a sink interface (Postgres sink added in M4); wrapper that measures latency around every provider call.
- Acceptance: each provider call produces exactly one log record with attempts, status, tokens and latency (verified with a fake sink, no database needed); cost math matches a fixture price table.
- Acceptance command: `bun --bun vitest run tests/unit/pricing.test.ts tests/unit/logger.test.ts`

**M4. Docker Compose, migrations, embedding tables, Postgres log sink (2 h)**
- Tasks: `docker-compose.yml` with `pgvector/pgvector:pg17` on port 5433, healthcheck `pg_isready`, optional `ollama` profile for Linux; `migrations/001_init.sql`; `src/db/migrate.ts`; template rendering in `embeddingTables.ts`; scripts `db:migrate`, `db:embeddings`; Postgres sink for `obs/logger.ts`; `vitest.config.ts` gets `unit` and `integration` projects. From here on, local `bun run test` needs `docker compose up -d`; integration tests fail fast with a clear message if `DATABASE_URL` is unset.
- Acceptance: running `bun run db:migrate && bun run db:embeddings` twice produces no error and no change; `SELECT extversion FROM pg_extension WHERE extname='vector'` returns at least 0.8; inserting a 1024 float vector into `emb_nomic_embed_text` fails; each provider call through the Postgres sink creates exactly one `llm_calls` row. Index check: seed one `documents` row and 2000 `chunks` rows (FK), then 2000 fake vectors from `fake.ts`, all with the same `chunk_config` (so the planner cannot prefer the `_cfg` btree); run `ANALYZE emb_nomic_embed_text, chunks`; inside one transaction apply the same `SET LOCAL hnsw.*` statements as `vector.ts` and run `EXPLAIN ANALYZE` on the exact SELECT with a literal vector; the output contains `emb_nomic_embed_text_hnsw`. `enable_seqscan` stays untouched.
- Acceptance command: `bun --bun vitest run tests/integration/schema.test.ts tests/integration/llm-calls.test.ts`

**M5. CI workflow (1 h)**
- Tasks: `.github/workflows/ci.yml` per section 9 with `oven-sh/setup-bun`, `bun install --frozen-lockfile`, `bun run lint`, `bun run typecheck`, `bun run test` with a `pgvector/pgvector:pg17` service container and fake providers.
- Acceptance: the same commands pass locally with the CI environment; no secrets referenced. After I push (pushing stays with me), the GitHub run is green, checked with `gh run watch --exit-status` or in the GitHub UI.
- Acceptance command: `CHAT_PROVIDER=fake EMBED_PROVIDER=fake bun run lint && bun run typecheck && bun run test` against `docker compose`

**M6. Corpus fetch, Markdown loader, chunker (2 h)**
- Tasks: `scripts/fetch-corpus.sh` (sparse clone of `docs/` at a pinned SHA recorded in `.env.example` as `CORPUS_SHA`; prints the sha256 of the file list as its last output line); add `corpus/` to `.gitignore`; verify license and record it in README; `loadMarkdown.ts` (frontmatter removal, MDX component stripping, title from first heading); `chunk.ts` using `RecursiveCharacterTextSplitter.fromLanguage("markdown", { chunkSize, chunkOverlap, lengthFunction })` with a token based length function, heading path tracked per chunk.
- Acceptance: fetch is reproducible (same SHA gives same file list hash); no chunk exceeds `chunk_size` tokens; code fences are not split mid fence when they fit in one chunk. Unit tests use fixture Markdown with headings, a code fence and an MDX tag, asserting chunk count, max tokens, heading paths and stripped JSX.
- Acceptance command: `bun --bun vitest run tests/unit/chunk.test.ts tests/unit/load-markdown.test.ts && [ "$(./scripts/fetch-corpus.sh | tail -n1)" = "$(./scripts/fetch-corpus.sh | tail -n1)" ]`

**M7. Idempotent ingest (2 h)**
- Tasks: `ingest.ts` and `cli/ingest.ts`; batch embedding (32 texts per call); transaction per document; empty corpus directory exits with a clear message.
- Acceptance: second run on an unchanged corpus makes zero embedding calls (checked in `llm_calls`); editing one fixture file re-embeds only that document's changed chunks. Integration test with fake embeddings (hash based deterministic 768 floats) asserts counts after run 1, run 2 and after a file edit.
- Acceptance command: `bun --bun vitest run tests/integration/ingest.test.ts`

**M8. Retrieval (1.5 h)**
- Tasks: `vector.ts`; `pgRetriever.ts` extending `BaseRetriever` from `@langchain/core/retrievers` and returning `Document` objects with `chunk_id`, `source_path`, `distance` in metadata.
- Acceptance: `bun run retrieve "how do I read env vars" --k 5` prints chunk IDs, sources and distances; filter by `chunk_config` always returns k rows when enough chunks exist. Integration test: the fake embedding of a query equals a known chunk's embedding, so that chunk ranks first.
- Acceptance command: `bun --bun vitest run tests/integration/retrieval.test.ts && bun run retrieve "how do I read env vars" --k 5`

**M9. Answer generation with citations and NOT_FOUND (2 h)**
- Tasks: `prompts.ts`, `schemas.ts` (`Answer = { found: boolean, answer: string, citations: number[] }`), `answer.ts`; context format `[chunk:123] (source_path > heading)\n<content>`; post validation that every citation is in the provided context; `rag_requests` logging.
- Acceptance: `bun run ask "<question>"` prints answer and citations; a question about a nonexistent feature returns `found=false`; invented citation IDs are dropped and logged as `invalid_output`. Unit tests: mocked chat returning citations outside the context; mocked chat returning `found=false`; prompt snapshot.
- Acceptance command: `bun --bun vitest run tests/unit/answer.test.ts && bun run ask "<question>"`

**M10. Golden dataset (3 h, me, not an agent)**
- Tasks: write 45 questions per section 6; `dataset.ts` with Zod schema and `bun run eval:validate`, which checks that every evidence span exists verbatim (after normalization) in the referenced source file. The validator and its unit test (broken span fixture) may be written by an agent; the questions may not.
- Acceptance: validator passes; split is 15 dev and 30 test; dataset hash recorded.
- Acceptance command: `bun run eval:validate`

**M11. Retrieval metrics and runner (2 h)**
- Tasks: `relevance.ts`, `metrics.ts`, `runner.ts`, `cli/eval.ts eval:retrieval`.
- Acceptance: writes `results/<timestamp>_baseline_retrieval.json` with per question and aggregate metrics, git SHA, dataset hash, corpus SHA, model tags. Unit tests use hand computed examples for Recall@k, MRR, nDCG@k (including no relevant hit and duplicate span coverage) and keyword coverage.
- Acceptance command: `bun --bun vitest run tests/unit/metrics.test.ts && bun run eval:retrieval --config experiments/baseline.json`

**M12. Answer evals with LLM-as-a-judge (2 h)**
- Tasks: `judge.ts`, `eval/prompts.ts`, `eval:answers`; abstention metrics for negative questions; judge self consistency run on a 10 percent sample.
- Acceptance: one answer eval run on the dev split completes and writes results; judge outputs validate against the Zod schema; failed validations are counted, not silently dropped. Unit tests on mocked judge responses and aggregation.
- Acceptance command: `bun --bun vitest run tests/unit/judge.test.ts && bun run eval:answers --config experiments/baseline.json --split dev`

**M13a. Experiment runner and README report (1.5 h)**
- Tasks: experiment file format per section 7; `cli/experiment.ts`; `cli/report.ts` renders tables between `<!-- results:start -->` and `<!-- results:end -->`; `bun run report --check` exits non zero if README differs from what `results/` generates.
- Acceptance: report output is deterministic; snapshot test from fixture result files.
- Acceptance command: `bun --bun vitest run tests/unit/report.test.ts`

**M13b. Run phase 1 experiments (1.5 h, me, mostly compute time)**
- Tasks: run `exp-chunk-size`, `exp-k`, `exp-embed-model` on the test split; regenerate README.
- Acceptance: one result file per configuration; README table generated, not hand typed.
- Acceptance command: `bun run experiment experiments/exp-chunk-size.json && bun run experiment experiments/exp-k.json && bun run experiment experiments/exp-embed-model.json && bun run report && bun run report --check`

### Phase 2: Advanced retrieval and agent, still local (about 12 h, 1.5 days)

**M14. Full text and hybrid search (2 h)**
- Tasks: `fts.ts` (`websearch_to_tsquery('english', $q)`, `ts_rank_cd`), `hybrid.ts` with reciprocal rank fusion (k = 60) in SQL:
  ```sql
  WITH v AS (
    SELECT id, row_number() OVER (ORDER BY dist) AS r FROM (
      SELECT e.chunk_id AS id, e.embedding <=> $1::vector AS dist
      FROM emb_nomic_embed_text e WHERE e.chunk_config = $2
      ORDER BY e.embedding <=> $1::vector LIMIT 50) s
  ), f AS (
    SELECT id, row_number() OVER (ORDER BY rank DESC) AS r FROM (
      SELECT c.id, ts_rank_cd(c.tsv, q) AS rank
      FROM chunks c, websearch_to_tsquery('english', $3) q
      WHERE c.chunk_config = $2 AND c.tsv @@ q
      ORDER BY rank DESC LIMIT 50) s
  )
  SELECT id, sum(1.0 / (60 + r)) AS score
  FROM (SELECT * FROM v UNION ALL SELECT * FROM f) u
  GROUP BY id ORDER BY score DESC LIMIT $4;
  ```
- Acceptance: retrieval results written for `vector`, `fts`, `hybrid`. RRF unit test on two ranked lists; integration test where FTS finds an exact identifier that vector search misses on fixtures.
- Acceptance command: `bun --bun vitest run tests/unit/rrf.test.ts tests/integration/hybrid.test.ts && bun run experiment experiments/exp-search-mode.json`

**M15. Query rewriting (1.5 h)**
- Tasks: `rewrite.ts` with structured output `{ rewritten: string }`; logged as operation `rewrite`; fallback to the original query on invalid output.
- Acceptance: eval run with rewrite on versus off; added latency per request reported.
- Acceptance command: `bun --bun vitest run tests/unit/rewrite.test.ts && bun run experiment experiments/exp-rewrite.json`

**M16. Query expansion (1.5 h)**
- Tasks: `expand.ts` with `{ queries: string[] }` (max 3), retrieve per query, fuse with RRF, deduplicate.
- Acceptance: eval run with expansion on versus off.
- Acceptance command: `bun --bun vitest run tests/unit/expand.test.ts && bun run experiment experiments/exp-expand.json`

**M17. LLM reranking (2 h)**
- Tasks: `rerank.ts`: retrieve top 20, one structured call returning `{ ranking: { chunk_id: number, score: number }[] }` over truncated chunks, keep top 5; unknown IDs ignored; missing IDs appended in original order.
- Acceptance: eval run with rerank on versus off, including added latency and tokens; unit test with invalid and missing IDs.
- Acceptance command: `bun --bun vitest run tests/unit/rerank.test.ts && bun run experiment experiments/exp-rerank.json`

**M18a. LangGraph agent graph (2 h)**
- Tasks: `graph.ts` with `StateGraph` and `Annotation` from `@langchain/langgraph`; nodes `agent` (calls `ChatProvider.generate` with tool `search_docs(query, k)`) and `tools` (executes the retriever); conditional edge on tool calls; hard limit of 4 tool calls; final answer through the same `Answer` schema. LangChain chat models are not used; messages are our own types.
- Acceptance: scripted fake chat that calls the tool twice then answers produces a valid `Answer`; a fake that loops forever is stopped at 4 tool calls; each step logged as `agent_step`.
- Acceptance command: `bun --bun vitest run tests/integration/agent.test.ts`

**M18b. Agent CLI and trace (1 h)**
- Tasks: `--pipeline agent` flag in `cli/ask.ts`; printed tool call trace (query, k, returned chunk IDs per step).
- Acceptance: the trace shows each tool call and the final cited answer against real Ollama.
- Acceptance command: `bun run ask --pipeline agent "<question>"`

**M19. Phase 2 evaluation and documentation (2 h)**
- Tasks: run the agent experiment on the test split with the same judge; extend README tables; write "what did not help".
- Acceptance: each technique has a row with delta versus baseline; README states plainly where a technique made results worse or added latency without gain.
- Acceptance command: `bun run experiment experiments/exp-agent.json && bun run report && bun run report --check`

### Phase 3 (OPTIONAL, built last): Anthropic adapter

**M20. Anthropic adapter (2 h)**
- Tasks: list `@anthropic-ai/sdk@1.7.0` for approval (re-check the current version then) and extend `tests/unit/deps.test.ts` with it; `anthropic.ts` implementing `ChatProvider` only (tools mapped to Anthropic tool format, structured output via tool use or the SDK's structured output feature, whichever is verified in the official docs at that time); `CHAT_PROVIDER=anthropic`, `ANTHROPIC_MODEL=claude-haiku-5-5`; embeddings stay on Ollama via `EMBED_PROVIDER=ollama`.
- Acceptance: the full pipeline and agent run with `CHAT_PROVIDER=anthropic` with zero changes outside `src/llm/` and `src/config.ts`. A single `LLMProvider` with `embed` would make this impossible, because Anthropic offers no embeddings endpoint; that is why the interface is split from M2 on (ADR 0003). Unit tests use a mocked SDK client; no real key in CI.
- Acceptance command: `bun --bun vitest run tests/unit/anthropic.test.ts && git diff --stat <M19 commit> -- src ':!src/llm' ':!src/config.ts'` (diff output must be empty)

**M21. Local versus Claude comparison (1.5 h)**
- Tasks: run answer generation for the 30 test questions with `qwen2.5:7b` and with `claude-haiku-5-5`; judge both with the same judge (`gemma3:12b`); compute judge scores, p50/p95 latency and cost per 100 questions from `llm_calls`.
- Budget: prepaid credit of 5 EUR with auto reload disabled; the run aborts when its estimated cost exceeds `MAX_RUN_COST_USD`.
- Acceptance: results file and README section including the judge limitation statement.
- Acceptance command: `bun run experiment experiments/exp-provider.json && bun run report && bun run report --check`

### Phase 4 (OPTIONAL learning goal): Cloud deployment

Recommendation: GCP. App container on Cloud Run (free tier request and compute quota), PostgreSQL with pgvector in a Docker container on an always free `e2-micro` VM, generation through the phase 3 Anthropic adapter, query embeddings through `nomic-embed-text` in an Ollama sidecar of the Cloud Run service on CPU. Justification: free tiers offer no GPU and roughly 1 GB RAM VMs, which cannot run 7b or 12b chat models, while a 768 dim embedding model on CPU and precomputed corpus embeddings are feasible. Assumption: Cloud Run free quota and sidecar support suffice for demo traffic; verify current free tier terms before starting.

**M22a. Postgres VM (2 h)**
- Tasks: `e2-micro` VM with the `pgvector/pgvector:pg17` container, firewall closed to the internet, restore of a `pg_dump` from local ingest, monthly budget alert at 1 EUR.
- Acceptance: database reachable only from inside the VPC.
- Acceptance command: `pg_isready -h <VM external IP>` fails from outside, and `psql -c 'select 1'` succeeds from inside the VPC

**M22b. Cloud Run app (2 h)**
- Tasks: container image for the app plus Ollama sidecar with `nomic-embed-text`; direct VPC egress to the VM; `ANTHROPIC_API_KEY` from Secret Manager; minimal HTTP endpoint `/ask`.
- Acceptance: public URL answers a question with citations.
- Acceptance command: `curl -s "<service URL>/ask?q=..."` returns JSON with citations

---

## 6. Eval design

### Building the golden dataset
- Read a docs section, close it, then write the question in my own words as a Bun user would ask it. Copy the evidence span (one sentence or line, at most 15 words) verbatim from the source afterwards.
- Format (`eval/golden.jsonl`):
  ```json
  {"id":"q007","split":"test","category":"direct_fact","question":"Which file does Bun load environment variables from without any extra flag?","answerable":true,"evidence":[{"source":"docs/runtime/env.md","span":"Bun reads the following files automatically"}],"keywords":[".env",".env.local"],"reference_answer":"Bun loads .env, .env.$NODE_ENV and .env.local automatically."}
  ```
  (Content above illustrates the format only; real entries come from the pinned corpus.)
- Categories and counts (45 total):
  | Category | Count | Description |
  |---|---|---|
  | direct_fact | 15 | One fact from one page |
  | how_to | 10 | Procedure, command or code snippet |
  | multi_document | 8 | Needs evidence from two or more pages |
  | vocabulary_mismatch | 5 | Question uses words that do not appear in the docs |
  | negative | 7 | Plausible but not covered by the docs; expected `NOT_FOUND` |
- Split: 15 dev (used for prompt tuning and debugging), 30 test (only for reported numbers). Stratified by category.

### Relevance definition
A retrieved chunk is relevant for question q if its `source_path` equals an evidence `source` and its normalized content (lowercase, collapsed whitespace, Markdown markup removed) contains the normalized `span`. This keeps labels valid across chunk sizes and embedding models. Negative questions are excluded from retrieval metrics.

### Metric definitions
Let E_q be the evidence spans of q, R_q^k the top k retrieved chunks, rank(i) the position starting at 1.

- **Recall@k** = |{e ∈ E_q : some chunk in R_q^k covers e}| / |E_q|, averaged over answerable questions.
- **MRR@k** = (1/|Q|) · Σ_q 1/rank_q, where rank_q is the position of the first relevant chunk in R_q^k, and the term is 0 if none.
- **nDCG@k**: rel_i = 1 if chunk i covers at least one span not covered by chunks 1..i−1, else 0. DCG@k = Σ_{i=1..k} rel_i / log2(i+1). IDCG@k = Σ_{i=1..min(k,|E_q|)} 1 / log2(i+1). nDCG@k = DCG@k / IDCG@k.
- **Keyword coverage@k** = |{w ∈ K_q : w occurs case insensitively in concat(R_q^k)}| / |K_q|.
- **Judge scores**: accuracy, completeness, relevance on a 1 to 5 scale, mean per dimension over answerable questions.
- **Abstention accuracy** = share of negative questions answered with `found=false`.
- **False abstention rate** = share of answerable questions answered with `found=false`.
- **Citation validity** = share of `found=true` answers whose citations are non empty and all inside the provided context.

### Judge prompt draft (`src/eval/prompts.ts`)
```
System:
You are a strict evaluator of answers about the Bun JavaScript runtime documentation.
You get a QUESTION, a REFERENCE ANSWER written by a human, the CONTEXT that was given to the
assistant, and the ASSISTANT ANSWER. Judge only the ASSISTANT ANSWER.

Rules:
1. Use the REFERENCE ANSWER and CONTEXT as ground truth. Do not use outside knowledge.
2. Write your reasoning first, at most 3 sentences, then the scores.
3. Longer answers do not earn higher scores. Penalize content not supported by the CONTEXT.
4. Scale for each criterion:
   accuracy:     5 = every claim correct; 3 = minor error; 1 = main claim wrong or unsupported
   completeness: 5 = covers all points of the reference; 3 = misses one important point; 1 = misses most
   relevance:    5 = answers exactly the question; 3 = partly off topic; 1 = does not address it
Return JSON matching the schema.

User:
QUESTION: {{question}}
REFERENCE ANSWER: {{reference_answer}}
CONTEXT: {{context}}
ASSISTANT ANSWER: {{answer}}
```
Zod schema:
```ts
export const JudgeResult = z.object({
  reasoning: z.string().max(600),
  accuracy: z.number().int().min(1).max(5),
  completeness: z.number().int().min(1).max(5),
  relevance: z.number().int().min(1).max(5),
});
```

### Rules against data contamination and judge bias
- Questions are written before any retrieval or answer output is seen; the dataset file is frozen once M10 ends, its sha256 is stored in every result file, and later edits create a new dataset version that invalidates old comparisons.
- No LLM writes or paraphrases golden questions or reference answers. If an exception ever happens, the entry is tagged `"origin":"llm"` and reported separately.
- Tuning (prompts, k, thresholds) only on dev; README numbers only from test.
- Pretraining contamination: Bun is likely in the models' training data. A closed book baseline (no context) is run once; answers are only counted as grounded if citations are valid.
- Judge model `gemma3:12b` differs in family from generator `qwen2.5:7b` to reduce self preference. Temperature 0, fixed seed where supported.
- Judge never sees which pipeline produced an answer; answers are shuffled per run.
- Self consistency: 10 percent of answers judged twice; report the share of identical scores.
- Human calibration: I score 20 test answers myself with the same rubric before reading judge scores and report agreement (share of scores within 1 point).

---

## 7. Experiment matrix

Fixed for all runs unless varied: test split, generator `qwen2.5:7b`, judge `gemma3:12b`, temperature 0, dataset hash, corpus SHA.

| Order | Experiment | Variable | Values | Fixed | Eval type |
|---|---|---|---|---|---|
| 1 | exp-chunk-size | chunk size (cl100k tokens), overlap 12.5 percent | 256, 512, 1024 | nomic, k=5, vector | retrieval |
| 2 | exp-k | top k | 3, 5, 10 | best chunk size, nomic | retrieval, then answers for best and baseline |
| 3 | exp-embed-model | embedding model | nomic-embed-text, mxbai-embed-large | chunk size 256 or 512 only (mxbai context is 512 tokens), best k | retrieval |
| 4 | exp-baseline-answers | none | best phase 1 config | | answers |
| 5 | exp-search-mode | search | vector, fts, hybrid (RRF) | best phase 1 config | retrieval |
| 6 | exp-rewrite | query rewrite | off, on | best search mode | retrieval + answers |
| 7 | exp-expand | query expansion | off, 3 queries | best search mode | retrieval + answers |
| 8 | exp-rerank | LLM rerank | off, top 20 to 5 | best search mode | retrieval + answers |
| 9 | exp-agent | pipeline | baseline, agent | best phase 2 retrieval as tool | answers |
| 10 (opt.) | exp-provider | generator | qwen2.5:7b, claude-haiku-5-5 | judge gemma3:12b | answers |
| (opt.) | exp-fts-config | tsvector config | english, simple | hybrid | retrieval |

One variable at a time (greedy). The greedy order is a limitation and is stated in the README.

Experiment file (`experiments/exp-chunk-size.json`):
```json
{
  "id": "exp-chunk-size",
  "base": { "embed": "nomic-embed-text", "k": 5, "search": "vector", "split": "test" },
  "vary": { "chunkConfig": ["md-256-32", "md-512-64", "md-1024-128"] },
  "evals": ["retrieval"]
}
```

Result file (`results/2026-11-20T10-00_exp-chunk-size_md-512-64.json`) contains: `experimentId`, `config`, `gitSha`, `datasetSha256`, `corpusSha`, model tags plus digests from `ollama show --modelfile` or `/api/tags`, `startedAt`, aggregate metrics, per question metrics, KPI block (section 8), `runId` linking to `llm_calls`.

---

## 8. Observability

Per model call (`llm_calls`): run_id, request_id, operation, provider, model, input and output tokens, token source (usage or estimate), latency ms, attempts, status, cost estimate.
Per RAG request (`rag_requests`): end to end latency, pipeline, retrieved chunk IDs, found flag.
Console: one structured JSON line per request at `LOG_LEVEL=info`; prompts and contexts only at `debug`.

KPI query (`src/obs/kpis.sql`):
```sql
SELECT pipeline,
       count(*)                                                        AS requests,
       percentile_cont(0.5)  WITHIN GROUP (ORDER BY total_latency_ms) AS p50_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY total_latency_ms) AS p95_ms
FROM rag_requests WHERE run_id = $1 GROUP BY pipeline;

SELECT operation, model,
       avg(input_tokens)  AS avg_in,
       avg(output_tokens) AS avg_out,
       sum(cost_usd) * 100.0 / count(DISTINCT request_id) AS cost_per_100_requests,
       avg((attempts > 1)::int)          AS retry_rate,
       avg((status <> 'ok')::int)        AS failure_rate
FROM llm_calls WHERE run_id = $1 GROUP BY operation, model;
```

README KPIs, generated by `bun run report` from result files:
- p50 and p95 end to end latency per pipeline (with hardware stated).
- Average input and output tokens per question; share of estimated token counts.
- Cost per 100 questions (0 for local; real for phase 3).
- Retry rate and invalid structured output rate.
- Abstention accuracy and false abstention rate.
- Latency and token overhead of each phase 2 technique relative to baseline.

Latency numbers from local runs depend on model load state: the runner sends one warm up request per model and excludes it.

---

## 9. Test and CI strategy

| Layer | Runs where | What | Model calls |
|---|---|---|---|
| Unit | local + CI | chunking, hashing, metrics, relevance, RRF, retry, structured output repair, pricing, report rendering | none or `fake.ts` |
| Integration | local + CI | migrations, ingest idempotency, vector and hybrid SQL, EXPLAIN index check, answer pipeline, agent loop | `fake.ts` (deterministic hash embeddings) and recorded responses from `tests/fixtures/llm/` |
| Dependency check | local + CI | `tests/unit/deps.test.ts`: every planned package imports and runs under Bun | none |
| Model check and spike | local only | M0 Ollama model check, `scripts/spike-ollama.ts` | real Ollama |
| Evals and experiments | local only | `eval:*`, `experiment` | real Ollama (phase 3: Anthropic) |

- Command: `bun run test` (baseline script: `bun --bun vitest run`, which forces the Bun runtime, plus `bun test ./.claude/hooks` for the hook tests). Lint is Biome via `bun run lint`.
- `vitest.config.ts`: the baseline only excludes `.claude/**`; M4 adds projects `unit` and `integration`. From M4 on, local `bun run test` needs `docker compose up -d`; integration tests read `DATABASE_URL` and fail fast with a clear message if it is unset.
- Mocking: model access is only reachable through `ChatProvider` and `EmbeddingProvider`, so tests inject fakes via `registry.ts` with `CHAT_PROVIDER=fake`. No HTTP mocking of the OpenAI SDK needed, except one adapter test that passes a stub `fetch` to the `OpenAI` client to verify request mapping.
- Fake embeddings: sha256 of the text expanded to `dim` floats and L2 normalized, so identical text gives identical vectors and tests are deterministic.
- CI (`.github/workflows/ci.yml`):
  ```yaml
  jobs:
    test:
      runs-on: ubuntu-latest
      services:
        db:
          image: pgvector/pgvector:pg17
          env: { POSTGRES_USER: rag, POSTGRES_PASSWORD: rag, POSTGRES_DB: rag_test }
          ports: ["5432:5432"]
          options: >-
            --health-cmd "pg_isready -U rag" --health-interval 5s --health-retries 10
      env:
        DATABASE_URL: postgres://rag:rag@localhost:5432/rag_test
        CHAT_PROVIDER: fake
        EMBED_PROVIDER: fake
      steps:
        - uses: actions/checkout@v4
        - uses: oven-sh/setup-bun@v2
          with: { bun-version: 1.4.4 }
        - run: bun install --frozen-lockfile
        - run: bun run lint
        - run: bun run typecheck
        - run: bun run test
  ```
- Secrets: `ANTHROPIC_API_KEY` exists only in local `.env` (gitignored) from my personal account, never an employer key, never in GitHub secrets, never in CI. `config.ts` refuses to start with `CHAT_PROVIDER=anthropic` when `CI=true`. The secret and PII gate is `scripts/privacy-scan.ts`, run by `scripts/git-hooks/` on every commit; no separate secret scanner is used.

---

## 10. Architecture decision records

1. **ADR 0001: Own provider interface instead of a framework abstraction.** The interface is a few dozen lines, makes retries, token accounting and cost logging explicit and testable, and shows I understand what an LLM call involves. Alternatives: Vercel AI SDK (excluded), LangChain chat models (adds an abstraction layer whose behavior I would have to debug), calling the SDK directly in the pipeline (no swap, no central logging).
2. **ADR 0002: pgvector instead of a dedicated vector database.** I already run PostgreSQL 17, and keeping chunks, vectors, full text index and logs in one database allows hybrid search and KPI queries in one SQL statement with transactional ingest. Alternatives: Chroma (excluded), Qdrant or Weaviate (extra service and sync logic, better at very large scale which this corpus does not need).
3. **ADR 0003: Separate ChatProvider and EmbeddingProvider.** Anthropic has no embeddings API, so one combined interface would force pipeline changes or a fake implementation in phase 3. Alternatives: single `LLMProvider` with `embed` throwing for Anthropic (hidden runtime failure), an embeddings proxy inside the Anthropic adapter (couples unrelated vendors).
4. **ADR 0004: OpenAI compatible endpoint as the default adapter.** The `openai` package against Ollama's `/v1` endpoint gives one code path for local and any OpenAI compatible host, and is the setup most local model tooling documents. Alternatives: the `ollama` npm client (native features such as `keep_alive`, but vendor specific), raw `fetch` (more code, no typed responses).
5. **ADR 0005: One embeddings table per model with fixed vector(N).** Fixed dimensions let HNSW indexes be used without casts and make dimension mismatches fail at insert time, while chunks stay shared across models. Alternatives: untyped `vector` column with partial expression indexes per model (queries must repeat the exact cast or silently lose the index), one table per model and chunk config (table explosion).
6. **ADR 0006: Golden labels are evidence spans, not chunk IDs.** Chunk IDs change with every chunk size and re-ingest, so labels tied to them would make chunking experiments incomparable. Alternatives: chunk ID labels per config (relabeling for every experiment), LLM generated relevance labels (circular evaluation).

---

## 11. README outline

1. Title, one line summary, CI badge.
2. What it does: question in, cited answer or `NOT_FOUND` out; screenshot of `bun run ask` output.
3. Architecture: diagram from section 1, link to `docs/adr/`.
4. Quickstart: prerequisites (Bun 1.4.4, Docker, Ollama), `brew services start ollama`, `ollama pull ...`, `docker compose up -d`, `bun install`, `bun run db:migrate && bun run db:embeddings`, `./scripts/fetch-corpus.sh`, `bun run ingest`, `bun run ask "..."`.
5. Hardware and model profiles: default (32 GB, `qwen2.5:7b` + `gemma3:12b`), light (16 GB, `qwen2.5:3b` + `llama3.1:8b`); 8 GB is not supported for the judge.
6. Corpus and license: source repo, pinned SHA, license, attribution.
7. Evaluation method: dataset categories, split, relevance definition, metric formulas (link to `docs/PLAN.md` section 6).
8. Results (generated between markers):
   | Config | Recall@5 | MRR@5 | nDCG@5 | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|
   | baseline (phase 1 best) | | | | | | | | | | | | |
   | hybrid | | | | | | | | | | | | |
   | + rewrite | | | | | | | | | | | | |
   | + expansion | | | | | | | | | | | | |
   | + rerank | | | | | | | | | | | | |
   | agent | | | | | | | | | | | | |
   | claude-haiku-5-5 (opt.) | | | | | | | | | | | | |
   Plus separate tables for chunk size, k and embedding model.
9. KPIs: latency, tokens, cost, retry and invalid output rates.
10. Limitations and learnings: small dataset (45 questions, confidence is low for small deltas); local judge noise and measured self consistency; greedy one variable at a time search; Bun docs likely in pretraining data; token counts partly estimated with a non native tokenizer; latency measured on one laptop; which techniques did not help and why.
11. How this repo was built: Claude Code generated code and tests under my review; milestone plan in `docs/PLAN.md`.
12. Roadmap: optional phases 3 and 4.

---

## 12. Risks and pitfalls

| Risk | Countermeasure |
|---|---|
| Chunking splits code fences or loses heading context | Markdown aware splitter, heading path prefixed to chunk text and tsvector, unit tests on fixtures with fences |
| Chunk sizes measured in characters instead of tokens | Token based `lengthFunction`, assertion that no chunk exceeds the configured token size |
| Embedding input silently truncated (mxbai context is 512 tokens) | Embedding model experiment only at chunk sizes up to 512; ingest warns when a chunk exceeds the model context from `config/models.json` |
| Missing nomic task prefixes reduce retrieval quality | Prefixes applied inside the adapter based on `kind`; unit test on the adapter request |
| Embedding dimension mismatch | Fixed `vector(N)` per table, startup check of `atttypmod` against config, insert fails on wrong length |
| Filtered HNSW search returns fewer than k rows | `hnsw.iterative_scan = relaxed_order` (pgvector >= 0.8) and integration test asserting k rows |
| Index not used, silent sequential scan | `EXPLAIN ANALYZE` assertion in the integration test |
| Small model ignores instructions (citations, NOT_FOUND) | Structured output with Zod, post validation of citation IDs, repair retry, false abstention and citation validity tracked as metrics |
| Ollama structured output differs per model or version | M1 spike documents behavior per model; fallback to `json_object` plus Zod repair |
| Zod 4 incompatible with `openai/helpers/zod` | M1 check (b) on both chat models; fallback (c): `z.toJSONSchema` passed manually plus own `safeParse` validation and up to 2 repair retries |
| Double retries (SDK plus own) multiply latency | `maxRetries: 0` in SDK clients, own `retry.ts` only |
| Vitest silently runs on Node instead of Bun | `bun run test` uses `bun --bun vitest run`; `deps.test.ts` asserts the Bun runtime |
| Judge noise and self preference | Different model family, rubric anchors, reasoning first, self consistency sample, human calibration on 20 answers |
| Eval bias from tuning on test data | Dev and test split, dataset hash in results, dataset frozen after M10 |
| Lexical overlap favors FTS in evals | Questions written in own words, vocabulary mismatch category |
| Rate limits and cost (phase 3) | Retry on 429 with backoff, concurrency 2, `MAX_RUN_COST_USD` abort, prepaid credit without auto reload |
| Memory pressure with two chat models loaded | Check with `ollama ps` in M0; run judge after generation instead of interleaving if needed |
| Corpus changes upstream | Pinned SHA, file list hash in result files |
