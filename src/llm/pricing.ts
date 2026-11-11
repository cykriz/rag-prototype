// Cost estimate from config/pricing.json (docs/PLAN.md M3). Prices are USD per 1M tokens; local models cost 0.
import { z } from 'zod';
import pricingJson from '../../config/pricing.json';
import type { Usage } from './types';

const Price = z.object({
  inputPerMTok: z.number().nonnegative(),
  outputPerMTok: z.number().nonnegative(),
});
const Pricing = z.object({ models: z.record(z.string(), Price) });
export type Price = z.infer<typeof Price>;
export type PricingTable = z.infer<typeof Pricing>;

export const pricing = Pricing.parse(pricingJson);

/** Throws for an unpriced model: a silent 0 would defeat the run cost limit (docs/PLAN.md M21). */
export function priceFor(table: PricingTable, model: string): Price {
  const price = table.models[model];
  if (!price) {
    throw new Error(`No price for model "${model}" in config/pricing.json`);
  }
  return price;
}

export function costUsd(price: Price, usage: Usage | undefined): number {
  if (!usage) {
    return 0;
  }
  return (usage.inputTokens * price.inputPerMTok + usage.outputTokens * price.outputPerMTok) / 1_000_000;
}
