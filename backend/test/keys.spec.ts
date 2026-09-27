import { createHash } from 'crypto';
import { KEY_PREFIX, keyDigest, newKey } from '../src/keys/keys.service';
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
