# Spike results

## M0: Model check (2026-11-11)

Environment: Ollama 0.40.15 (native, macOS), Apple M5, 32 GB unified memory, Bun 1.4.4. No `OLLAMA_*`
environment variables set on the server, so defaults apply (context 32768, default max loaded models).

### Models

| Model | ID | Disk | `ollama ps` size (ctx 32768) | `ollama ps` size (ctx 8192) | Processor | Eval rate tok/s (median of 3) |
|---|---|---|---|---|---|---|
| qwen2.5:7b | 845dbda0ea48 | 4.7 GB | 6.6 GB | not measured | 100% GPU | **16.1** (16.14, 16.28, 15.86) |
| gemma3:12b | b92b218582c5 | 8.2 GB | 8.3 GB | 7.3 GB | 100% GPU | **9.6** (9.65, 9.52, 9.63) |
| nomic-embed-text | 0a109f422b47 | 274 MB | not loaded | not loaded | n/a | n/a |
| mxbai-embed-large | 468836162de7 | 669 MB | not loaded | not loaded | n/a | n/a |

All four models were pulled and verified with `ollama pull`.

### Method

- Warm-up: `ollama run --keepalive 15m <model> "Say ok"` once per model, then three timed runs with
  `ollama run --verbose --keepalive 15m <model> "Explain in about 150 words what a JavaScript runtime does."`
  (91 to 178 output tokens per run). This differs from the plan's `"Say ok"` because a 2 token answer
  makes the eval rate mostly noise.
- Co-residency: load one chat model, then the other, and check `ollama ps`. Repeated in both orders at
  the default context, and once via `/api/generate` with `num_ctx: 8192` for both models.
- `tests/unit/deps.test.ts` compiles a fully empty `StateGraph`, as the plan states. That works with
  `@langchain/langgraph@1.7.3`.

### Verdict

| Criterion | Result |
|---|---|
| `qwen2.5:7b` ≥ 15 tok/s | **pass** (16.1 tok/s, small margin) |
| Both chat models loaded together | **fail**: loading either model evicts the other. This happens in both orders, at ctx 32768 and at ctx 8192 (≈ 12 to 15 GB combined, with about 45 % of system memory free) |
| Profile | default (`qwen2.5:7b` + `gemma3:12b`), with generation and judging run in sequence |

Cause: not the loaded-models limit. A re-run with the server started as
`OLLAMA_MAX_LOADED_MODELS=2 ollama serve` (verified in the process environment) still evicts
`qwen2.5:7b` when `gemma3:12b` loads. [Assumption] The eviction comes from Ollama's scheduler estimate of
available GPU memory, not from a lack of physical RAM: the combined model size is well below 32 GB.
Decision: keep the default profile and run generation and judging as separate batches.

Consequence for later milestones: run answer generation first, then judging, as a separate batch, not
interleaved per question (risk table in `docs/PLAN.md` §12). Otherwise every switch reloads a model.
Gemma's 9.6 tok/s is acceptable for a batch judge.

## M1: Ollama capability spike (2026-11-11)

Environment: Ollama 0.40.15, `openai@7.55.0`, `zod@4.6.7`, Bun 1.4.4. Script: `bun scripts/spike-ollama.ts`
(exit 0). Each check ran once at `temperature: 0`, `seed: 42`. The test schema has the same shape as the later
M9 answer (`found`, `answer`, `citations: int[]`).

| Check | qwen2.5:7b | gemma3:12b | Embeddings |
|---|---|---|---|
| (a) `response_format` json_schema, hand-written schema | pass | pass | n/a |
| (b) `zodResponseFormat` + `chat.completions.parse` (Zod 4) | pass | pass | n/a |
| (c) `z.toJSONSchema` as json_schema + own `safeParse` | pass | pass | n/a |
| (d) `tools` with one function (`search_docs`) | pass, args `{"query": …}` | not checked (judge needs no tools) | n/a |
| (e) `embeddings.create` dimension | n/a | n/a | pass: nomic-embed-text 768, mxbai-embed-large 1024 |
| (f) `usage` present | pass (prompt and completion tokens) | pass | pass (`prompt_tokens`) |
| (g) `json_object` + `safeParse` | pass | pass | n/a |

Chosen structured output path: **(b) `zodResponseFormat` + `chat.completions.parse`** for both models
(generator `qwen2.5:7b` and judge `gemma3:12b`). The Zod 4 helper works as is, so the fallbacks (c) and (g)
are not needed. They stay available as the documented repair path from `docs/PLAN.md` M1.

Notes:
- Embeddings were requested with `encoding_format: 'float'`. The SDK default (base64) was not tested, so
  the M2 adapter sets `float` explicitly.
- Ollama returns `usage` on chat and embedding responses, so `tiktoken_estimate` (M3) is only a fallback.
- The first call per model includes the model load time (about 6 to 13 s). Later structured calls took
  about 1.8 s (qwen) and 3.3 s (gemma). This confirms the M0 decision to run generation and judging as separate batches.
