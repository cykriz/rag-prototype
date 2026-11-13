import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { configDefaults, defineConfig } from 'vitest/config';

// Bun skips its .env auto-loading for `bun --bun vitest`, so load the repo's .env here for every Vitest call,
// whatever the cwd. Variables already set (shell, CI) win; without a .env (CI) nothing happens.
const envFile = fileURLToPath(new URL('.env', import.meta.url));
if (existsSync(envFile)) {
  process.loadEnvFile(envFile);
}

export default defineConfig({
  test: {
    // .claude/hooks tests use bun:test and run via `bun test ./.claude/hooks`.
    exclude: [...configDefaults.exclude, '.claude/**'],
    projects: [
      {
        extends: true,
        test: { name: 'unit', include: ['tests/unit/**/*.test.ts', 'scripts/**/*.test.ts'] },
      },
      {
        extends: true,
        test: {
          name: 'integration',
          include: ['tests/integration/**/*.test.ts'],
          globalSetup: ['tests/integration/globalSetup.ts'],
          // Files share one database; schema.test.ts drops and recreates the schema.
          fileParallelism: false,
        },
      },
    ],
  },
});
