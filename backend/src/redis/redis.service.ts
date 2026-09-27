import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import Redis from 'ioredis';
import { config } from '../config';

/** Keys and usage are shared with the model-service through Redis (see model-service/src/dragonfly/auth.py). */
export const KEYS_SET = 'dragonfly:keys';
export const usageKey = (digest: string, day: string) => `dragonfly:usage:${digest}:${day}`;

@Injectable()
export class RedisService implements OnModuleDestroy {
  private readonly log = new Logger(RedisService.name);
  readonly client: Redis;

  constructor() {
    this.client = new Redis(config.redisUrl, { lazyConnect: false, maxRetriesPerRequest: 3 });
    this.client.on('error', (e) => this.log.warn(`redis: ${e.message}`));
  }

  async onModuleDestroy() {
    await this.client.quit();
  }
}
