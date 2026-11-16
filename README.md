# RAG Prototype

Local, offline RAG over the Bun documentation. Design, milestones and schema: [docs/PLAN.md](docs/PLAN.md).

## Corpus and license

The corpus is the Bun documentation (`docs/` in [oven-sh/bun](https://github.com/oven-sh/bun)) at the commit pinned as `CORPUS_SHA` in `.env.example`. `./scripts/fetch-corpus.sh` downloads it into `corpus/`, which is gitignored; this repository contains no corpus files.

License, checked on 2026-11-15 at commit `b6f9cfa0e820b025916e50651b377ee1dff4d173` (main, 2026-11-14, after tag `bun-v1.4.4`): `LICENSE.md` states "Bun itself is MIT-licensed." It names no separate license for the documentation, neither do `docs/README.md` and `docs/project/license.mdx`, and GitHub reports the repository license as "Other". Assumption: the documentation is covered by that MIT statement. Copyright remains with the Bun authors. The corpus is used locally only; the evaluation dataset quotes short spans from it, each with its source path.

## Results

Method: phase 1 varies one setting at a time in a fixed greedy order: chunk size, then k, then the embedding model. At each step the winner is chosen on the dev split, and the tables below show the test split only. Dev has 15 questions, 13 of them answerable. The selection rule ranks by Recall@k first and MRR@k second. A difference of at most one dev question (1/13) on a metric counts as a tie, and a tie goes to the cheaper option: smaller chunks, smaller k, the default embedding model. With so few dev questions this choice is noisy. For example, md-1024-128 won on dev but scores below md-512-64 on test. The selected phase 1 config is md-1024-128 with nomic-embed-text, k=5 and vector search: the `exp-k-answers` row.

Phase 2 starts with the search mode (`exp-search-mode`) at that config: vector search, full text search (`websearch_to_tsquery('english', …)` ranked by `ts_rank_cd`) and hybrid search, which fuses the top max(k, 50) of both with reciprocal rank fusion (k = 60) in SQL. The selection rule is the same; for ties the cost order is fts < vector < hybrid, because full text search needs no embedding call and hybrid needs both. On dev, vector and hybrid tie (Recall@5 0.69 for both; hybrid's MRR@5 lead of 0.05 is within 1/13), so vector search stays. On test, hybrid is ahead (Recall@5 0.80 against 0.72): the same small dev split problem as above. Full text search alone falls far behind because `websearch_to_tsquery` joins every word of a question with AND: 8 of 13 dev questions and 12 of 25 test questions match no chunk at all. From this step on every result file carries `schemaVersion` 2; only the retrieval layout changed: each hit stores a `score`, higher is better within one mode (vector: 1 − cosine distance), instead of the vector `distance`.

Next comes query rewriting (`exp-rewrite`), at that config with vector search. Before retrieval, the chat model turns the question into one search query, using temperature 0 and seed 42. The prompt lets it make vague wording concrete and add Bun terms. It must keep identifiers exactly as written, must not answer the question and must not name APIs the question does not imply. The rewritten query is used for both the embedding and the full text query, while the answer prompt keeps the original question. If the output is still invalid after two repairs, retrieval falls back to the original question; no run needed this. The answers decide, because the rewrite costs a model call on every request. Retrieval breaks a tie only, and a full tie goes to off. On dev, off wins: accuracy 4.69 against 4.17, well outside 1/13. Retrieval agrees, with Recall@5 0.69 against 0.58. The rewrites read well but flatten specific terms, so three dev questions lost their evidence and one gained it. On test, rewriting is ahead (accuracy 4.91 against 4.77, Recall@5 0.76 against 0.72), the same small dev split problem again, so the pipeline stays without rewriting. The rewrite call itself takes about 0.9 s at the median. The end-to-end p50 grows by 1.4 s on test (8264 against 6870 ms) and by 1.0 s on dev. Result files now carry `schemaVersion` 3. Each question records the query that retrieval used and whether it fell back. Answers files also report the latency percentiles of each call type.

Then query expansion (`exp-expand`), at the same config with rewriting off. The chat model writes up to three extra search queries for the question, using temperature 0 and seed 42. It may rephrase, use Bun terms and split a question that asks several things, but must keep identifiers exactly as written and must not answer. Retrieval searches the question and every expansion separately, fuses the lists with reciprocal rank fusion (depth max(k, 50) per list) and keeps the top k. Expansions that repeat the question or each other are dropped. If the output is still invalid after two repairs, retrieval uses the question alone; no run needed this. The same rule as for rewriting decides: answers first, retrieval only on a tie, a full tie goes to off. On dev, off wins: accuracy 4.69 against 4.46, well outside 1/13. Retrieval agrees, with Recall@5 0.69 against 0.58. The expansions mostly reword the question, and the fused lists push evidence out of the top 5: four dev questions lost evidence and one gained it. On test, expansion is ahead again (accuracy 4.92 against 4.77, Recall@5 0.76 against 0.72), so the pipeline stays without expansion, as it did for rewriting. The expand call takes 1.1 to 1.7 s at the median. On dev the end-to-end p50 grows by 2.7 s (16994 against 14266 ms); on test it is 1.2 s lower with expansion (8979 against 10184 ms), which is load noise on the laptop rather than a saving. Result files now carry `schemaVersion` 4. Each question records the expansions that were searched, how many duplicates were dropped and whether the stage fell back.

Then LLM reranking (`exp-rerank`), at the same config with rewriting and expansion off. Vector search fetches the top 20 chunks. The chat model scores each of them from 0 to 10 for how much it helps answer the question, using temperature 0 and seed 42. Each chunk is cut to 1200 characters, enough to show 88 % of the dev evidence spans. The five best-scored chunks go into the context. IDs the model invents are ignored, and chunks it leaves out are appended in search order. If the output is still invalid after two repairs, the search order stays; no run needed this. In practice the model scores only a handful of chunks: on average about 16.5 of the 20 are left out and appended. The same rule decides: answers first, retrieval only on a tie, a full tie goes to off. On dev, reranking wins: accuracy 4.85 against 4.69, a gap of 0.15, outside 1/13. The whole gap comes from a single question, which went from 3 to 5. Retrieval agrees, with Recall@5 0.77 against 0.69 and MRR@5 0.71 against 0.40. One dev question gained evidence and none lost any. On test, the answers come out even (accuracy 4.78 against 4.77, completeness 4.43 against 4.36). Retrieval improves again, with Recall@5 0.78 against 0.72 and MRR@5 0.73 against 0.61. Two test questions gained evidence and none lost any, and one of them is now answered instead of abstained (false abstention 0.08 against 0.12). The gain is expensive. The rerank call reads about 5800 input tokens and writes about 80 output tokens, and takes 28.6 s at the median. The end-to-end p50 on test grows by 31 s (43961 against 12681 ms), and tokens per question roughly triple. The dev rule picks reranking, but on test it bought better ranking, not better answers, at about 30 s per question. So the later milestones deliberately depart from the rule and run without reranking; the stage stays available as `--rerank` and `rerank: "on"`. Result files now carry `schemaVersion` 5. Each question records the candidates, the reranked order with the model's scores, the counts of unknown, duplicate and missing IDs, and whether the stage fell back.

Last, the agent (`exp-agent`), a LangGraph loop around the same chat model. Instead of one fixed search, the model gets a `search_docs(query, k)` tool backed by vector search at the same config (md-1024-128, k = 5 by default, at most 10). It may call the tool up to four times before it has to answer. Its final reply must be a valid cited answer over the chunks its searches returned. Otherwise one structured answer call with the normal answer prompt runs over those chunks (the answer fallback). The judge sees exactly those chunks as context. Only the agent arm ran, on test. It is compared with the newest run of the same config without the agent (the rerank-off row, same shuffle seed), so there is no dev selection here. The agent is slightly worse: accuracy 4.65 against 4.77 and completeness 4.30 against 4.36. Abstention accuracy drops from 1.00 to 0.80, because it answered one of the five unanswerable questions. In exchange it abstained wrongly a little less often (false abstention 0.08 against 0.12). The agent never used its freedom. It searched exactly once on every question, mostly with k = 5, and never reached the limit. 6 of 30 answers (20 %) needed the answer fallback. In effect it is the baseline with an extra model turn. That turn costs 3.0 s at the median (15634 against 12681 ms; `exp-k-answers` measured the same baseline at 6879 ms, so the latency gap is noisy) and 42 % more tokens per question (3913 against 2746). The agent is therefore not adopted. Result files now carry `schemaVersion` 6, with each question's tool call trace, the tool call count, whether the limit was reached and whether the answer fell back.

### What did not help

- **Hybrid search:** tied with vector on dev and was ahead on test retrieval only (Recall@5 +0.08). It never had an answers run, so vector search stayed.
- **Query rewriting:** lost clearly on dev (accuracy 4.17 against 4.69) but was ahead on test (+0.14). With 13 answerable dev questions, the selection signal is weak in both directions. Not adopted.
- **Query expansion:** same picture: it lost on dev (4.46 against 4.69) and was ahead on test (+0.14). Not adopted.
- **LLM reranking:** better ranking (Recall@5 +0.06, nDCG +0.12), but answers stayed even (+0.01 accuracy). It added 31 s at p50 and about three times the tokens. Latency without gain, so it is off.
- **Agent:** worse answers (accuracy −0.12, abstention accuracy 1.00 → 0.80), 3 s more at p50 and 42 % more tokens. It never searched more than once.

Latency numbers come from a single Apple M-series laptop (10 cores, 32 GB unified memory) running Ollama on the GPU, with models loaded and one warm-up request per model.

<!-- results:start -->
<!-- Generated by `bun run report` from results/; do not edit by hand. -->

| Config | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (md-512-64, nomic-embed-text, k=5, vector) | 0.76 | 0.62 | 0.63 | 0.92 | 4.70 | 4.35 | 4.91 | 1.00 | 5271 | 7600 | 1930 | 0.00 |
| exp-k-answers (md-1024-128, nomic-embed-text, k=5, vector) | – | – | – | – | 4.77 | 4.36 | 4.82 | 1.00 | 6879 | 10811 | 2746 | 0.00 |

### exp-chunk-size: chunkConfig

Fixed: nomic-embed-text, k=5, vector.

| chunkConfig | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| md-256-32 | 0.74 | 0.52 | 0.55 | 0.84 | – | – | – | – | – | – | – | – |
| md-512-64 | 0.76 | 0.62 | 0.63 | 0.92 | – | – | – | – | – | – | – | – |
| md-1024-128 | 0.72 | 0.61 | 0.60 | 0.90 | – | – | – | – | – | – | – | – |

### exp-k: k

Fixed: md-1024-128, nomic-embed-text, vector.

| k | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.70 | 0.61 | 0.59 | 0.86 | – | – | – | – | – | – | – | – |
| 5 | 0.72 | 0.61 | 0.60 | 0.90 | – | – | – | – | – | – | – | – |
| 10 | 0.80 | 0.61 | 0.62 | 0.94 | – | – | – | – | – | – | – | – |

### exp-embed-model: embed

Fixed: md-256-32, k=5, vector.

| embed | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mxbai-embed-large | 0.72 | 0.51 | 0.54 | 0.92 | – | – | – | – | – | – | – | – |
| nomic-embed-text | 0.74 | 0.52 | 0.55 | 0.84 | – | – | – | – | – | – | – | – |

### exp-search-mode: search

Fixed: md-1024-128, nomic-embed-text, k=5.

| search | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fts | 0.48 | 0.35 | 0.38 | 0.52 | – | – | – | – | – | – | – | – |
| hybrid | 0.80 | 0.71 | 0.69 | 0.98 | – | – | – | – | – | – | – | – |
| vector | 0.72 | 0.61 | 0.60 | 0.90 | – | – | – | – | – | – | – | – |

### exp-rewrite: rewrite

Fixed: md-1024-128, nomic-embed-text, k=5, vector.

| rewrite | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | 0.72 | 0.61 | 0.60 | 0.90 | 4.77 | 4.36 | 4.82 | 1.00 | 6870 | 10806 | 2746 | 0.00 |
| on | 0.76 | 0.72 | 0.69 | 0.94 | 4.91 | 4.55 | 5.00 | 1.00 | 8264 | 12853 | 2986 | 0.00 |

### exp-expand: expand

Fixed: md-1024-128, nomic-embed-text, k=5, vector.

| expand | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | 0.72 | 0.61 | 0.60 | 0.90 | 4.77 | 4.36 | 4.82 | 1.00 | 10184 | 18987 | 2746 | 0.00 |
| on | 0.76 | 0.63 | 0.63 | 0.94 | 4.92 | 4.58 | 5.00 | 1.00 | 8979 | 11740 | 3026 | 0.00 |

### exp-rerank: rerank

Fixed: md-1024-128, nomic-embed-text, k=5, vector.

| rerank | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | 0.72 | 0.61 | 0.60 | 0.90 | 4.77 | 4.36 | 4.82 | 1.00 | 12681 | 19134 | 2746 | 0.00 |
| on | 0.78 | 0.73 | 0.72 | 0.92 | 4.78 | 4.43 | 4.83 | 1.00 | 43961 | 53160 | 8716 | 0.00 |

### exp-agent: pipeline

Fixed: md-1024-128, nomic-embed-text, k=5, vector.

| pipeline | Recall@k | MRR@k | nDCG@k | KW cov. | Accuracy | Completeness | Relevance | Abstention acc. | p50 ms | p95 ms | Tokens/q | Cost/100 q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| agent | – | – | – | – | 4.65 | 4.30 | 4.87 | 0.80 | 15634 | 28941 | 3913 | 0.00 |

### Phase 2 deltas vs. baseline

| Technique | Baseline | Δ Recall@k | Δ nDCG@k | Δ Accuracy | Δ Completeness | Δ Abstention acc. | Δ p50 ms | Δ Tokens/q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-search-mode: search=fts | exp-search-mode: search=vector | -0.24 | -0.21 | – | – | – | – | – |
| exp-search-mode: search=hybrid | exp-search-mode: search=vector | +0.08 | +0.09 | – | – | – | – | – |
| exp-rewrite: rewrite=on | exp-rewrite: rewrite=off | +0.04 | +0.09 | +0.14 | +0.18 | +0.00 | +1394 | +240 |
| exp-expand: expand=on | exp-expand: expand=off | +0.04 | +0.04 | +0.14 | +0.22 | +0.00 | -1205 | +280 |
| exp-rerank: rerank=on | exp-rerank: rerank=off | +0.06 | +0.12 | +0.01 | +0.07 | +0.00 | +31280 | +5970 |
| exp-agent: pipeline=agent | exp-rerank: rerank=off | – | – | -0.12 | -0.06 | -0.20 | +2953 | +1167 |

Each row minus its baseline: the same experiment with the technique off, or else the newest run of the same config with it off (named in the Baseline column).

### KPIs

| Config | In tok/q | Out tok/q | Est. token share | Retry rate | Invalid output rate | False abst. |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (md-512-64, nomic-embed-text, k=5, vector) | 1884 | 46 | 0.00 | 0.07 | 0.00 | 0.08 |
| exp-k-answers (md-1024-128, nomic-embed-text, k=5, vector) | 2700 | 46 | 0.00 | 0.00 | 0.00 | 0.12 |
| exp-rewrite: rewrite=off | 2700 | 46 | 0.00 | 0.00 | 0.00 | 0.12 |
| exp-rewrite: rewrite=on | 2922 | 64 | 0.00 | 0.00 | 0.00 | 0.12 |
| exp-expand: expand=off | 2700 | 46 | 0.00 | 0.00 | 0.00 | 0.12 |
| exp-expand: expand=on | 2955 | 71 | 0.00 | 0.00 | 0.00 | 0.04 |
| exp-rerank: rerank=off | 2700 | 46 | 0.00 | 0.00 | 0.00 | 0.12 |
| exp-rerank: rerank=on | 8593 | 122 | 0.00 | 0.00 | 0.00 | 0.08 |
| exp-agent: pipeline=agent | 3820 | 93 | 0.00 | 0.00 | 0.00 | 0.08 |

Test split only, latest run per config from a clean checkout. Metrics are the `all` group of each result file (human and llm reference answers are reported separately there); `–` = not run or not recorded. Tokens/q, Cost/100 q and the KPI rates count the answer pipeline without the judge. Dataset `89f4a8a4`, corpus `b6f9cfa0`.
<!-- results:end -->

## Limitations

- **Small dataset.** 45 questions (15 dev, 30 test), so differences of a few hundredths are within noise.
- **Greedy search.** Settings are tuned one at a time (chunk size → k → embedding model → search mode), and interactions are not explored. The result is not necessarily the best combination.
- **Embedding model comparison.** mxbai-embed-large has a 512-token context, so it was only compared at md-256-32 (the cheaper of the ≤ 512 configs, tied on dev). It was not tried at md-1024-128, the chunk size chosen in the first step.
- **Latency.** Measured on one laptop and depends on model load state and hardware.
