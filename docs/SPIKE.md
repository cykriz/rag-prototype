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
