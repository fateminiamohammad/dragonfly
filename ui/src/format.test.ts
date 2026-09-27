import { describe, expect, it } from 'vitest';
import { headline, pct, topOptions, totalsByDay } from './format';

describe('format', () => {
  it('formats percentages', () => {
    expect(pct(0.5)).toBe('50.0%');
    expect(pct(1)).toBe('100%');
  });

  it('headlines each answer type', () => {
    expect(headline({ type: 'choice', choice: 'refund', confidence: 1 })).toBe('refund');
    expect(headline({ type: 'noul', noul: 0.8, confidence: 0.6 })).toBe('yes (80.0% yes)');
    expect(headline({ type: 'score', score: 2.5, confidence: 0.5, probabilities: { '0': 0, '1': 0, '2': 0.5, '3': 0.5 } })).toBe('2.50 / 3');
  });

  it('sorts options and applies score legends', () => {
    const top = topOptions({ type: 'score', score: 1, confidence: 1, probabilities: { '0': 0.1, '1': 0.9 }, legend: { '0': 'bad', '1': 'good' } });
    expect(top).toEqual([
      { label: '1 · good', p: 0.9 },
      { label: '0 · bad', p: 0.1 },
    ]);
  });

  it('totals usage per day across keys', () => {
    const rows = [
      { day: 'd2', keyId: 'a', keyName: 'a', requests: 2, questions: 4, tokens: 10 },
      { day: 'd2', keyId: 'b', keyName: 'b', requests: 1, questions: 1, tokens: 5 },
    ];
    expect(totalsByDay(['d1', 'd2'], rows).map((t) => t.requests)).toEqual([0, 3]);
  });
});
