CREATE TABLE IF NOT EXISTS {{table}} (
  chunk_id      bigint PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  chunk_config  text   NOT NULL,                 -- denormalized for filtered ANN search
  content_hash  char(64) NOT NULL,               -- allows reuse of vectors for unchanged chunks
  embedding     vector({{dim}}) NOT NULL
);
CREATE INDEX IF NOT EXISTS {{table}}_hnsw ON {{table}}
  USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS {{table}}_cfg ON {{table}} (chunk_config);
