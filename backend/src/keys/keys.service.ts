import { Injectable, Logger, NotFoundException, OnApplicationBootstrap } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { createHash, randomBytes } from 'crypto';
import { IsNull, Repository } from 'typeorm';
import { ApiKey } from '../entities/api-key.entity';
import { KEYS_SET, RedisService } from '../redis/redis.service';

export const KEY_PREFIX = 'df_';

export function newKey(): string {
  return KEY_PREFIX + randomBytes(32).toString('base64url');
}

/** Same digest the model-service computes: sha256 hex of the full key. */
export function keyDigest(key: string): string {
  return createHash('sha256').update(key).digest('hex');
}

export interface KeyView {
  id: string;
  name: string;
  prefix: string;
  createdAt: Date;
  revokedAt: Date | null;
}

const view = (k: ApiKey): KeyView => ({ id: k.id, name: k.name, prefix: k.prefix, createdAt: k.createdAt, revokedAt: k.revokedAt });

@Injectable()
export class KeysService implements OnApplicationBootstrap {
  private readonly log = new Logger(KeysService.name);

  constructor(
    @InjectRepository(ApiKey) private readonly keys: Repository<ApiKey>,
    private readonly redis: RedisService,
  ) {}

  /** Postgres is the source of truth: rebuild the Redis set at startup (covers a flushed or new Redis). */
  async onApplicationBootstrap() {
    await this.sync().catch((e) => this.log.error(`key sync failed: ${e.message}`));
  }

  async sync() {
    const active = await this.keys.find({ where: { revokedAt: IsNull() } });
    const pipe = this.redis.client.multi().del(KEYS_SET);
    if (active.length) pipe.sadd(KEYS_SET, ...active.map((k) => k.digest));
    await pipe.exec();
    this.log.log(`synced ${active.length} active API keys to redis`);
  }

  async list(userId: string): Promise<KeyView[]> {
    return (await this.keys.find({ where: { userId }, order: { createdAt: 'DESC' } })).map(view);
  }

  /** Returns the plaintext key exactly once. */
  async create(userId: string, name: string): Promise<KeyView & { key: string }> {
    const key = newKey();
    const saved = await this.keys.save(
      this.keys.create({ name, userId, prefix: key.slice(0, KEY_PREFIX.length + 6), digest: keyDigest(key), revokedAt: null }),
    );
    await this.redis.client.sadd(KEYS_SET, saved.digest);
    return { ...view(saved), key };
  }

  async revoke(userId: string, id: string): Promise<KeyView> {
    const key = await this.keys.findOne({ where: { id, userId } });
    if (!key) throw new NotFoundException();
    if (!key.revokedAt) {
      key.revokedAt = new Date();
      await this.keys.save(key);
      await this.redis.client.srem(KEYS_SET, key.digest);
    }
    return view(key);
  }

  async digestsFor(userId: string): Promise<{ id: string; name: string; prefix: string; digest: string }[]> {
    return this.keys.find({ where: { userId }, select: ['id', 'name', 'prefix', 'digest'] });
  }
}
