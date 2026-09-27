import { Controller, Get, Query } from '@nestjs/common';
import { AuthUser, CurrentUser } from '../auth/auth.guard';
import { KeysService } from '../keys/keys.service';
import { RedisService, usageKey } from '../redis/redis.service';

export interface UsageRow {
  day: string;
  keyId: string;
  keyName: string;
  requests: number;
  questions: number;
  tokens: number;
}

export function lastDays(n: number, now = new Date()): string[] {
  return Array.from({ length: n }, (_, i) => new Date(now.getTime() - i * 86_400_000).toISOString().slice(0, 10)).reverse();
}

/** Usage counters are written by the model-service into Redis per key and UTC day; this only reads them. */
@Controller('usage')
export class UsageController {
  constructor(
    private readonly keys: KeysService,
    private readonly redis: RedisService,
  ) {}

  @Get()
  async usage(@CurrentUser() user: AuthUser, @Query('days') daysParam?: string) {
    const days = lastDays(Math.min(Math.max(Number(daysParam) || 30, 1), 90));
    const keys = await this.keys.digestsFor(user.id);
    const pipe = this.redis.client.pipeline();
    for (const k of keys) for (const d of days) pipe.hgetall(usageKey(k.digest, d));
    const results = (await pipe.exec()) ?? [];
    const rows: UsageRow[] = [];
    let i = 0;
    for (const k of keys) {
      for (const day of days) {
        const [, h] = results[i++] as [Error | null, Record<string, string>];
        if (h && Object.keys(h).length) {
          rows.push({ day, keyId: k.id, keyName: `${k.name} (${k.prefix}…)`, requests: +h.requests || 0, questions: +h.questions || 0, tokens: +h.tokens || 0 });
        }
      }
    }
    return { days, rows };
  }
}
