#!/usr/bin/env bash
# Sparse, blobless clone of docs/ from oven-sh/bun at the pinned CORPUS_SHA into corpus/bun-docs (docs/PLAN.md M6).
# CORPUS_SHA comes from the environment, else from .env.example. Progress goes to stderr; the last stdout line is
# the sha256 of the file list (path and content hash of every file under docs/), so the same SHA prints the same
# line. A corpus already at the SHA needs no network; it is reset to a pristine checkout either way.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote="https://github.com/oven-sh/bun.git"
target="$repo_root/corpus/bun-docs"

if [ -z "${CORPUS_SHA:-}" ]; then
  CORPUS_SHA="$(sed -n 's/^CORPUS_SHA=//p' "$repo_root/.env.example" | head -n 1)"
fi
if ! [[ "$CORPUS_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "fetch-corpus: CORPUS_SHA must be a full 40-character lowercase commit SHA, got '$CORPUS_SHA'" >&2
  exit 1
fi
if ! command -v git >/dev/null; then
  echo "fetch-corpus: git is required" >&2
  exit 1
fi

current=""
if [ -d "$target/.git" ]; then
  current="$(git -C "$target" rev-parse -q --verify HEAD || true)"
fi

if [ "$current" = "$CORPUS_SHA" ]; then
  echo "fetch-corpus: corpus/bun-docs already at $CORPUS_SHA, no fetch" >&2
else
  echo "fetch-corpus: fetching docs/ of oven-sh/bun at $CORPUS_SHA" >&2
  rm -rf "$target"
  mkdir -p "$target"
  git -C "$target" init -q
  git -C "$target" remote add origin "$remote"
  # Non-cone: only docs/, not the upstream root files (their CLAUDE.md, package.json, ...).
  git -C "$target" sparse-checkout set --no-cone /docs/
  git -C "$target" fetch -q --depth 1 --filter=blob:none origin "$CORPUS_SHA"
fi
git -C "$target" -c advice.detachedHead=false checkout -q --force --detach "$CORPUS_SHA"
git -C "$target" clean -fdqx

cd "$target"
find docs -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -d ' ' -f 1
