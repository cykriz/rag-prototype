import { configDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // .claude/hooks tests use bun:test and run via `bun test ./.claude/hooks`.
    exclude: [...configDefaults.exclude, '.claude/**'],
  },
});
