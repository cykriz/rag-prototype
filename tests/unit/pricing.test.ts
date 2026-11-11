import { describe, expect, it } from 'vitest';

import { models } from '../../src/config';
import { FAKE_CHAT_MODEL, FAKE_JUDGE_MODEL } from '../../src/llm/fake';
import { costUsd, type PricingTable, priceFor, pricing } from '../../src/llm/pricing';

const FIXTURE: PricingTable = {
  models: {
    'paid-model': { inputPerMTok: 3, outputPerMTok: 15 },
    'free-model': { inputPerMTok: 0, outputPerMTok: 0 },
  },
};

describe('costUsd', () => {
  it('prices input and output tokens per million', () => {
    const price = priceFor(FIXTURE, 'paid-model');
    expect(costUsd(price, { inputTokens: 1000, outputTokens: 500 })).toBeCloseTo(0.0105, 12);
    expect(costUsd(price, { inputTokens: 1_000_000, outputTokens: 0 })).toBe(3);
  });

  it('is 0 for a free model and without usage', () => {
    expect(costUsd(priceFor(FIXTURE, 'free-model'), { inputTokens: 1000, outputTokens: 500 })).toBe(0);
    expect(costUsd(priceFor(FIXTURE, 'paid-model'), undefined)).toBe(0);
  });
});

describe('priceFor', () => {
  it('throws for an unpriced model instead of returning 0', () => {
    expect(() => priceFor(FIXTURE, 'unknown')).toThrow('No price for model "unknown"');
  });
});

describe('config/pricing.json', () => {
  it('prices every configured model and the fakes at 0', () => {
    const configured = [
      ...Object.values(models.profiles).flatMap((p) => [p.chat, p.judge]),
      ...Object.keys(models.embeddings),
      FAKE_CHAT_MODEL,
      FAKE_JUDGE_MODEL,
    ];
    for (const model of configured) {
      expect(priceFor(pricing, model), model).toEqual({ inputPerMTok: 0, outputPerMTok: 0 });
    }
  });
});
