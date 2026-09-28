import { BadRequestException, Body, Controller, Get, Header, NotFoundException, Param, Post, Query } from '@nestjs/common';
import { IsString, Length } from 'class-validator';
import { RedisService } from '../redis/redis.service';

/** The human-review plugin (model-service `dragonfly.essentials.human_review`) pushes unsure decisions here. */
export const PENDING = 'dragonfly:review:pending';
export const DONE = 'dragonfly:review:done';

export interface ReviewItem {
  id: string;
  created: number;
  model: string;
  state: unknown;
  question_id: string;
  question: { type: 'choice' | 'noul' | 'score'; instructions: unknown; criteria: unknown };
  answer: Record<string, unknown>;
  label?: string;
  reviewed_at?: number;
}

/** The option keys a reviewer can pick, in the same shape as the API's probabilities. */
export function optionKeys(q: ReviewItem['question']): string[] {
  if (q.type === 'noul') return ['false', 'true'];
  if (q.type === 'score') return (q.criteria as unknown[]).map((_, i) => String(i));
  return Object.keys((q.criteria as Record<string, unknown>) ?? {});
}

/** A reviewed item as one line of Dragonfly training JSONL (data.py format: a request plus a label per question). */
export function toTrainingRecord(item: ReviewItem) {
  const q = item.question;
  const label = q.type === 'noul' ? item.label === 'true' : q.type === 'score' ? Number(item.label) : item.label;
  return {
    state: item.state,
    questions: { [item.question_id]: { type: q.type, instructions: q.instructions, criteria: q.criteria, label } },
    _meta: { source: 'human-review', id: item.id, reviewed_at: item.reviewed_at },
  };
}

class LabelDto {
  @IsString()
  @Length(1, 200)
  label: string;
}

@Controller('review')
export class ReviewController {
  constructor(private readonly redis: RedisService) {}

  private async pending(limit = 500): Promise<{ raw: string; item: ReviewItem }[]> {
    const raws = await this.redis.client.lrange(PENDING, 0, limit - 1);
    return raws.map((raw) => ({ raw, item: JSON.parse(raw) as ReviewItem }));
  }

  @Get()
  async list(@Query('limit') limit?: string) {
    const items = await this.pending(Math.min(Math.max(Number(limit) || 50, 1), 200));
    return {
      pending: await this.redis.client.llen(PENDING),
      reviewed: await this.redis.client.llen(DONE),
      items: items.map(({ item }) => ({ ...item, options: optionKeys(item.question) })),
    };
  }

  @Post(':id')
  async label(@Param('id') id: string, @Body() dto: LabelDto) {
    const found = (await this.pending()).find((x) => x.item.id === id);
    if (!found) throw new NotFoundException('review item not found (already reviewed?)');
    if (!optionKeys(found.item.question).includes(dto.label)) {
      throw new BadRequestException(`label must be one of: ${optionKeys(found.item.question).join(', ')}`);
    }
    const done: ReviewItem = { ...found.item, label: dto.label, reviewed_at: Date.now() / 1000 };
    await this.redis.client.multi().lrem(PENDING, 1, found.raw).lpush(DONE, JSON.stringify(done)).exec();
    return { ok: true, id };
  }

  @Post(':id/skip')
  async skip(@Param('id') id: string) {
    const found = (await this.pending()).find((x) => x.item.id === id);
    if (!found) throw new NotFoundException('review item not found');
    await this.redis.client.lrem(PENDING, 1, found.raw);
    return { ok: true, id };
  }

  /** Reviewed items as training JSONL: feed to `dragonfly-train` or a specialist (active learning). */
  @Get('export')
  @Header('content-type', 'application/x-ndjson')
  @Header('content-disposition', 'attachment; filename="dragonfly-reviewed.jsonl"')
  async export() {
    const done = await this.redis.client.lrange(DONE, 0, -1);
    return done.map((raw) => JSON.stringify(toTrainingRecord(JSON.parse(raw) as ReviewItem))).join('\n') + '\n';
  }
}
