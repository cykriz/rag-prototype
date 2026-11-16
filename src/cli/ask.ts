// bun run ask "<question>" [--k 5] [--chunk md-512-64] [--embed <model>] [--search vector|fts|hybrid] [--rewrite]
// Retrieves, answers with citations or NOT_FOUND, logs the request (docs/PLAN.md M9).
import { randomUUID } from 'node:crypto';
import type { Sql } from 'postgres';
import type { Config } from '../config';
import { type AnswerResult, answer, buildAnswerDeps, RetrievalError } from '../rag/answer';
import { sourceLabel } from '../retrieval/pgRetriever';
import { QUESTION_FLAGS, runQuestionCli } from './common';
import { DB_DOWN, describeError, isDatabaseDown, unwrapStageError } from './errors';

const USAGE = `Usage: bun run ask "<question>" ${QUESTION_FLAGS}`;

/** stdout: answer, citations and the found flag; stderr: warnings about dropped or missing citations. */
export function formatAnswer(res: AnswerResult): { out: string; warnings: string[] } {
  const lines = [res.answer ?? ''];
  if (res.citations.length > 0) {
    lines.push('', 'Citations:');
    for (const c of res.citations) {
      lines.push(
        `  [chunk:${c.chunkId}] ${sourceLabel({ source_path: c.sourcePath, heading_path: c.headingPath })}`,
      );
    }
  }
  lines.push('', `found=${res.found}  request=${res.requestId}`);
  const warnings: string[] = [];
  if (res.droppedCitations.length > 0) {
    warnings.push(`dropped citations outside the context: ${res.droppedCitations.join(', ')}`);
  }
  if (res.found === true && res.citations.length === 0) {
    warnings.push('unverified: the answer has no valid citations');
  }
  return { out: lines.join('\n'), warnings };
}

/**
 * Message for an error from answer(). Unwraps RetrievalError and RewriteError first: the 404 hint names the embedding
 * model for retrieval failures, the chat model otherwise (rewrite and generate both run on it).
 */
export function describeAnswerError(err: unknown, config: Config, dbDown: boolean): string {
  if (dbDown) {
    return DB_DOWN;
  }
  const retrieval = err instanceof RetrievalError;
  const inner = unwrapStageError(err);
  return describeError(inner, {
    baseURL: config.baseURL,
    model: retrieval ? config.embedModel : config.chatModel,
  });
}

/** describeAnswerError after probing the database with the unwrapped error; shared by ask and eval:answers. */
export async function answerErrorMessage(err: unknown, config: Config, sql: Sql): Promise<string> {
  const dbDown = await isDatabaseDown(unwrapStageError(err), sql);
  return describeAnswerError(err, config, dbDown);
}

function main(argv: string[]): Promise<number> {
  return runQuestionCli(argv, USAGE, async ({ args, config, chunkConfig, sql }) => {
    let res: AnswerResult;
    try {
      const deps = buildAnswerDeps(sql, config, {
        chunkConfig,
        k: args.k,
        runId: `ask-${randomUUID()}`,
        pipeline: args.pipeline,
      });
      res = await answer(args.question, deps);
    } catch (err) {
      console.error(await answerErrorMessage(err, config, sql));
      return 1;
    }
    if (res.found === null) {
      console.error(
        `The model returned no usable answer after ${res.repairs} repairs (request ${res.requestId})`,
      );
      return 1;
    }
    const { out, warnings } = formatAnswer(res);
    console.log(out);
    for (const w of warnings) {
      console.error(w);
    }
    return 0;
  });
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
