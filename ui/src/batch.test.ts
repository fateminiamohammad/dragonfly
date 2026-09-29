import { describe, expect, it } from 'vitest';
import { buildQuestion, parseCsv } from './pages/Batch';

describe('batch page helpers', () => {
  it('parses quoted CSV', () => {
    expect(parseCsv('text,team\n"a, ""b""",x\r\nc,y')).toEqual([
      ['text', 'team'],
      ['a, "b"', 'x'],
      ['c', 'y'],
    ]);
  });
  it('builds the question for each type', () => {
    expect(buildQuestion('noul', 'urgent?', '')).toEqual({ type: 'noul', instructions: 'urgent?' });
    expect(buildQuestion('choice', 'team?', 'a, b')).toEqual({ type: 'choice', instructions: 'team?', criteria: { a: null, b: null } });
    expect(buildQuestion('score', 'mood?', 'low,high')).toEqual({ type: 'score', instructions: 'mood?', criteria: ['low', 'high'] });
  });
});
