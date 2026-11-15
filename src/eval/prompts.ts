// Judge prompt for the answer eval (docs/PLAN.md §6 "Judge prompt draft", M12).
import type { Message } from '../llm/types';

/** Fixed for reproducible runs (§6 "Rules against data contamination and judge bias"). */
export const JUDGE_TEMPERATURE = 0;
export const JUDGE_SEED = 42;

// §6 draft without "written by a human": 20 of the 45 reference answers carry origin "llm".
export const JUDGE_SYSTEM_PROMPT = `You are a strict evaluator of answers about the Bun JavaScript runtime documentation.
You get a QUESTION, a REFERENCE ANSWER, the CONTEXT that was given to the
assistant, and the ASSISTANT ANSWER. Judge only the ASSISTANT ANSWER.

Rules:
1. Use the REFERENCE ANSWER and CONTEXT as ground truth. Do not use outside knowledge.
2. Write your reasoning first, at most 3 sentences, then the scores.
3. Longer answers do not earn higher scores. Penalize content not supported by the CONTEXT.
4. Scale for each criterion:
   accuracy:     5 = every claim correct; 3 = minor error; 1 = main claim wrong or unsupported
   completeness: 5 = covers all points of the reference; 3 = misses one important point; 1 = misses most
   relevance:    5 = answers exactly the question; 3 = partly off topic; 1 = does not address it
Return JSON matching the schema.`;

export interface JudgeInput {
  question: string;
  referenceAnswer: string;
  /** The context text the assistant saw (AnswerResult.context). */
  context: string;
  answer: string;
}

/** The judge sees no pipeline, model or config: only these four fields. */
export function buildJudgeMessages(input: JudgeInput): Message[] {
  return [
    { role: 'system', content: JUDGE_SYSTEM_PROMPT },
    {
      role: 'user',
      content:
        `QUESTION: ${input.question}\n` +
        `REFERENCE ANSWER: ${input.referenceAnswer}\n` +
        `CONTEXT: ${input.context}\n` +
        `ASSISTANT ANSWER: ${input.answer}`,
    },
  ];
}
