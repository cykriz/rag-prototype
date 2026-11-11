// M0 dependency check: every planned package imports and does one offline call under the Bun runtime.
// Guards Bun compatibility on every later dependency bump.
import { Document } from '@langchain/core/documents';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { getEncoding } from 'js-tiktoken';
import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import pgvector from 'pgvector';
import postgres from 'postgres';
import { describe, expect, it } from 'vitest';
import { z } from 'zod';

describe('dependencies under Bun', () => {
  it('runs on the Bun runtime', () => {
    expect(typeof Bun).not.toBe('undefined');
  });

  it('@langchain/textsplitters splits a string', async () => {
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 10, chunkOverlap: 0 });
    const chunks = await splitter.splitText('alpha beta gamma delta epsilon');
    expect(chunks.length).toBeGreaterThan(1);
  });

  it('zod produces a JSON schema', () => {
    const schema = z.toJSONSchema(z.object({ a: z.string() }));
    expect(schema).toMatchObject({ type: 'object', properties: { a: { type: 'string' } } });
  });

  it('js-tiktoken encodes with cl100k_base', () => {
    expect(getEncoding('cl100k_base').encode('hello world').length).toBeGreaterThan(0);
  });

  it('@langchain/langgraph compiles an empty StateGraph', () => {
    const State = Annotation.Root({ count: Annotation<number>() });
    expect(new StateGraph(State).compile()).toBeDefined();
  });

  it('openai client instantiates with a custom baseURL', () => {
    const client = new OpenAI({ baseURL: 'http://localhost:11434/v1', apiKey: 'dummy', maxRetries: 0 });
    expect(client.baseURL).toBe('http://localhost:11434/v1');
  });

  it('openai/helpers/zod exports zodResponseFormat', () => {
    expect(typeof zodResponseFormat).toBe('function');
  });

  it('@langchain/core constructs a Document', () => {
    expect(new Document({ pageContent: 'x' }).pageContent).toBe('x');
  });

  it('postgres constructs a client without connecting', async () => {
    const sql = postgres('postgres://user:pass@localhost:5433/db');
    expect(typeof sql).toBe('function');
    await sql.end();
  });

  it('pgvector serializes a vector', () => {
    expect(pgvector.toSql([1, 2, 3])).toBe('[1,2,3]');
  });
});
