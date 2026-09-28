import { createHash } from 'crypto';
import { checkFlow } from '../src/flows/flows.controller';
import { KEY_PREFIX, keyDigest, newKey } from '../src/keys/keys.service';
import { optionKeys, toTrainingRecord } from '../src/review/review.controller';
import { csvLabel, parseCsv, toTrainingLines } from '../src/specialists/specialists.controller';
import { lastDays } from '../src/usage/usage.controller';

describe('API keys', () => {
  it('are prefixed, long and unique', () => {
    const a = newKey();
    const b = newKey();
    expect(a.startsWith(KEY_PREFIX)).toBe(true);
    expect(a.length).toBeGreaterThanOrEqual(40);
    expect(a).not.toEqual(b);
  });

  it('digest matches the model-service (sha256 hex of the full key)', () => {
    // model-service/src/dragonfly/auth.py: hashlib.sha256(key.encode()).hexdigest()
    expect(keyDigest('df_example')).toEqual(createHash('sha256').update('df_example').digest('hex'));
    expect(keyDigest('secret')).toEqual('2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b');
  });
});

describe('usage days', () => {
  it('lists the last n UTC days, oldest first', () => {
    expect(lastDays(3, new Date('2026-09-27T10:00:00Z'))).toEqual(['2026-09-25', '2026-09-26', '2026-09-27']);
  });
});


describe('review export', () => {
  const base = { id: 'x', created: 0, model: 'm', state: 'doc', question_id: 'q', answer: {} };
  it('offers the API option keys per question type', () => {
    expect(optionKeys({ type: 'noul', instructions: null, criteria: null })).toEqual(['false', 'true']);
    expect(optionKeys({ type: 'score', instructions: null, criteria: ['a', 'b', 'c'] })).toEqual(['0', '1', '2']);
    expect(optionKeys({ type: 'choice', instructions: null, criteria: { refund: null, billing: 'x' } })).toEqual(['refund', 'billing']);
  });
  it('exports labels in the training format (bool for noul, int for score)', () => {
    const noul = toTrainingRecord({ ...base, question: { type: 'noul', instructions: 'urgent?', criteria: null }, label: 'true' });
    expect(noul.questions.q.label).toBe(true);
    const score = toTrainingRecord({ ...base, question: { type: 'score', instructions: null, criteria: ['a', 'b'] }, label: '1' });
    expect(score.questions.q.label).toBe(1);
    expect(score._meta.source).toBe('human-review');
  });
});

describe('specialist uploads', () => {
  it('parses quoted CSV fields', () => {
    expect(parseCsv('text,label\n"a, ""quoted"" text",yes\r\nplain,no\n')).toEqual([
      ['text', 'label'],
      ['a, "quoted" text', 'yes'],
      ['plain', 'no'],
    ]);
  });
  it('maps labels per question type', () => {
    expect(csvLabel({ type: 'noul', instructions: 'x' }, 'Yes').label).toBe(true);
    expect(csvLabel({ type: 'choice', instructions: 'x', options: ['a', 'b'] }, 'b').label).toBe('b');
    expect(csvLabel({ type: 'score', instructions: 'x', options: ['low', 'high'] }, 'high').label).toBe(1);
    expect(csvLabel({ type: 'choice', instructions: 'x', options: ['a'] }, 'c').error).toBeDefined();
  });
  it('turns a CSV into training lines and refuses too little data', () => {
    const rows = Array.from({ length: 60 }, (_, i) => `text ${i},${i % 2 ? 'refund' : 'billing'}`);
    const lines = toTrainingLines({
      format: 'csv',
      data: ['text,label', ...rows].join('\n'),
      question: { type: 'choice', instructions: 'intent?', options: ['refund', 'billing'] },
    });
    expect(lines).toHaveLength(60);
    expect(JSON.parse(lines[1]).questions.q).toEqual({
      type: 'choice', instructions: 'intent?', criteria: { refund: null, billing: null }, label: 'refund',
    });
    expect(() => toTrainingLines({ format: 'jsonl', data: lines.slice(0, 10).join('\n') })).toThrow(/at least 50/);
    expect(() => toTrainingLines({ format: 'jsonl', data: '{"state":"x"}' })).toThrow(/bad rows/);
  });
});

describe('flows', () => {
  it('reports structural errors before saving', () => {
    expect(checkFlow({ steps: { a: { questions: { q: {} }, next: [{ to: 'a' }] } } })).toEqual([]);
    expect(checkFlow({})).toHaveLength(1);
    expect(checkFlow({ start: 'x', steps: { a: { next: [{ to: 'b' }] } } })).toEqual([
      'start step "x" is not defined',
      'step "a" needs questions',
      'step "a" goes to undefined step "b"',
    ]);
  });
});
