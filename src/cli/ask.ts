// bun run ask "<question>" [--k 5] [--chunk mdfm-1024-128] [--embed <model>] [--search vector|fts|hybrid]
//   [--rewrite | --expand] [--rerank] [--pipeline baseline|agent] [--style concise|complete]
// Retrieves (reranked with --rerank, M17), answers with citations or NOT_FOUND, logs the request (docs/PLAN.md M9).
// --pipeline agent (M18b): the LangGraph agent searches itself; its tool-call trace is printed before the answer.
// --style complete (M19c): the answer prompt that combines every relevant chunk; not with the agent.
import { randomUUID } from 'node:crypto';
import type { Sql } from 'postgres';
import { type AgentResult, buildAgentDeps, runAgent } from '../agent/graph';
import type { Config } from '../config';
import { type AnswerResult, answer, buildAnswerDeps, RetrievalError } from '../rag/answer';
import { ANSWER_PIPELINES, ANSWER_STYLES, MAX_TOOL_CALLS } from '../rag/pipeline';
import { sourceLabel } from '../retrieval/pgRetriever';
import { QUESTION_FLAGS, runQuestionCli } from './common';
import { DB_DOWN, describeError, isDatabaseDown, unwrapStageError } from './errors';

const USAGE = `Usage: bun run ask "<question>" ${QUESTION_FLAGS} [--pipeline ${ANSWER_PIPELINES.join('|')}] [--style ${ANSWER_STYLES.join('|')}]`;

/** One line per search_docs call (query, k, returned chunk IDs), then limit and fallback notes (M18b). */
export function formatAgentTrace(
  res: Pick<AgentResult, 'trace' | 'skippedToolCalls' | 'stoppedAtLimit' | 'answerFallback'>,
): string {
  const lines = res.trace.map((step, i) => {
    const n = `[${i + 1}]`;
    if (step.error !== null) {
      return `${n} search_docs error: ${step.error}`;
    }
    const requested =
      step.kRequested !== null && step.kRequested !== step.k ? ` (requested ${step.kRequested})` : '';
    const ids = step.chunkIds.length > 0 ? step.chunkIds.join(', ') : '(no results)';
    return `${n} search_docs(${JSON.stringify(step.query)}, k=${step.k}${requested}) → ${ids}`;
  });
  if (res.trace.length === 0) {
    lines.push('[0] no search');
  }
  if (res.stoppedAtLimit) {
    const skipped = res.skippedToolCalls > 0 ? `; ${res.skippedToolCalls} more not executed` : '';
    lines.push(`[limit] stopped after ${MAX_TOOL_CALLS} tool calls${skipped}`);
  }
  if (res.answerFallback) {
    lines.push('[answer] structured answer call over the retrieved chunks');
  }
  return lines.join('\n');
}

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
 * Message for an error from answer(). Unwraps RetrievalError and ChatStageError (rewrite, expand, rerank) first: the 404
 * hint names the embedding model for retrieval failures, the chat model otherwise (every chat stage and generate run on
 * it).
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
  return runQuestionCli(
    argv,
    USAGE,
    async ({ args, config, chunkConfig, sql }) => {
      let res: AnswerResult;
      const runId = `ask-${randomUUID()}`;
      try {
        if (args.agent) {
          const agentRes = await runAgent(
            args.question,
            buildAgentDeps(sql, config, { chunkConfig, k: args.k, search: args.pipeline.search, runId }),
          );
          console.log(`${formatAgentTrace(agentRes)}\n`);
          res = agentRes;
        } else {
          res = await answer(
            args.question,
            buildAnswerDeps(sql, config, {
              chunkConfig,
              k: args.k,
              runId,
              pipeline: args.pipeline,
              answerStyle: args.answerStyle,
            }),
          );
        }
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
    },
    { askFlags: true },
  );
}

if (import.meta.main) {
  process.exit(await main(process.argv.slice(2)));
}
