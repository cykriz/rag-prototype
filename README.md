# RAG Prototype

Local, offline RAG over the Bun documentation. Design, milestones and schema: [docs/PLAN.md](docs/PLAN.md).

## Corpus and license

The corpus is the Bun documentation (`docs/` in [oven-sh/bun](https://github.com/oven-sh/bun)) at the commit pinned as `CORPUS_SHA` in `.env.example`. `./scripts/fetch-corpus.sh` downloads it into `corpus/`, which is gitignored; this repository contains no corpus files.

License, checked on 2026-11-13 at commit `38afee6c84fcf48c0edb2466aa4b27d287a59c30` (tag `bun-v1.4.4`): `LICENSE.md` states "Bun itself is MIT-licensed." It names no separate license for the documentation, neither do `docs/README.md` and `docs/project/license.mdx`, and GitHub reports the repository license as "Other". Assumption: the documentation is covered by that MIT statement. Copyright remains with the Bun authors. The corpus is used locally only; the evaluation dataset quotes short spans from it, each with its source path.
