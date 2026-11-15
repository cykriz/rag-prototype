import { describe, expect, it } from 'vitest';
import { normalize } from '../../src/eval/normalize';
import { coverage } from '../../src/eval/relevance';

const SOURCE = 'guides/widget.md';

describe('coverage', () => {
  it('needs the same source path', () => {
    const chunk = { sourcePath: 'guides/other.md', content: 'Widgets live at example.com.' };
    expect(coverage([chunk], [{ source: SOURCE, span: 'Widgets live at example.com' }])).toEqual([[]]);
  });

  it('matches through Markdown markup and returns the covered span indices per chunk', () => {
    const chunks = [
      { sourcePath: SOURCE, content: 'Run `widget start` to open **https://example.com**.' },
      { sourcePath: SOURCE, content: 'Nothing relevant here.' },
    ];
    const evidence = [
      { source: SOURCE, span: 'not in any chunk' },
      { source: SOURCE, span: 'run widget start to open https://example.com' },
    ];
    expect(coverage(chunks, evidence)).toEqual([[1], []]);
  });

  it('matches a span across a plain line break', () => {
    const chunk = { sourcePath: SOURCE, content: 'The widget reads\nits config from example.com' };
    expect(coverage([chunk], [{ source: SOURCE, span: 'widget reads its config' }])).toEqual([[0]]);
  });

  it('still matches a chunk that starts at a real list item', () => {
    // Cut at a line start: the body strips "- " too, the chunk keeps it, the span never needs it.
    const chunk = { sourcePath: SOURCE, content: '- Run `widget start` on example.com\n- Then stop it' };
    const evidence = [
      { source: SOURCE, span: 'Run widget start on example.com' },
      { source: SOURCE, span: '- Run `widget start`' },
      { source: SOURCE, span: 'on example.com\n- Then stop it' },
    ];
    expect(coverage([chunk], evidence)).toEqual([[0, 1, 2]]);
  });

  describe('chunk cut mid-line before text that looks like a block marker', () => {
    // The body line is "Step 2. Then fetch …"; the chunker cut after "Step ", so the chunk starts with "2. ".
    // In the body "2." is mid-line text, not a list marker, and the validator checks spans against the body.
    const body = 'Intro text.\nStep 2. Then fetch https://example.com/api with `curl`.';
    const chunk = { sourcePath: SOURCE, content: '2. Then fetch https://example.com/api with `curl`.' };
    const span = '. Then fetch https://example.com/api';

    it('the span is valid against the body', () => {
      expect(normalize(body).includes(normalize(span))).toBe(true);
    });

    it('plain normalize() of the chunk strips "2. " and would miss the span', () => {
      expect(normalize(chunk.content).includes(normalize(span))).toBe(false);
    });

    it('coverage() normalizes the chunk as mid-line text and finds it', () => {
      expect(coverage([chunk], [{ source: SOURCE, span }])).toEqual([[0]]);
    });

    it('a span starting at the marker matches as well', () => {
      const evidence = [{ source: SOURCE, span: '2. Then fetch https://example.com/api' }];
      expect(coverage([chunk], evidence)).toEqual([[0]]);
    });
  });
});
