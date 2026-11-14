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
  content_hash    char(64) NOT NULL,                    -- sha256 of parsed title and body
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
