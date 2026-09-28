import { createHash } from 'crypto';
import { KEY_PREFIX, keyDigest, newKey } from '../src/keys/keys.service';
import { optionKeys, toTrainingRecord } from '../src/review/review.controller';
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
