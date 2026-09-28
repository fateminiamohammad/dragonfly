import { BadRequestException, Body, Controller, Delete, Get, Param, Post } from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import { Type } from 'class-transformer';
import { IsArray, IsIn, IsInt, IsOptional, IsString, Length, Matches, Max, Min, ValidateNested } from 'class-validator';
import { randomUUID } from 'crypto';
import { mkdir, writeFile } from 'fs/promises';
import { join } from 'path';
import { Roles } from '../auth/auth.guard';
import { config } from '../config';
import { callModel } from '../model/model.controller';
import { RedisService } from '../redis/redis.service';

/** The trainer worker (model-service `dragonfly.worker`) takes jobs from this queue. */
export const JOBS = 'dragonfly:train-jobs';
export const JOB_LIST = 'dragonfly:train-jobs:recent';
export const jobKey = (id: string) => `dragonfly:train-job:${id}`;
export const MIN_EXAMPLES = 50;
const RESERVED = ['auto', 'general', 'dragonfly-latest', 'jev-latest', 'kev-latest'];

type QType = 'choice' | 'noul' | 'score';

class QuestionDto {
  @IsIn(['choice', 'noul', 'score'])
  type: QType;

  @IsString()
  @Length(1, 2000)
  instructions: string;

  @IsOptional()
  @IsArray()
  @IsString({ each: true })
  options?: string[];
}

class CreateSpecialistDto {
  @Matches(/^[a-z0-9][a-z0-9-]{1,39}$/, { message: 'name: 2-40 lowercase letters, digits and dashes' })
  name: string;

  @IsString()
  @Length(3, 300)
  description: string;

  @IsIn(['S', 'M'])
  tier: 'S' | 'M';

  @IsIn(['csv', 'jsonl'])
  format: 'csv' | 'jsonl';

  @IsString()
  data: string;

  /** csv only: the question every row answers. Columns: text,label */
  @IsOptional()
  @ValidateNested()
  @Type(() => QuestionDto)
  question?: QuestionDto;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(10)
  epochs?: number;
}

/** RFC 4180 CSV: quoted fields, doubled quotes, commas and newlines inside quotes. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field);
      if (row.some((f) => f.trim())) rows.push(row);
      row = [];
      field = '';
    } else field += c;
  }
  row.push(field);
  if (row.some((f) => f.trim())) rows.push(row);
  return rows;
}

const TRUE = ['true', 'yes', 'y', '1'];
const FALSE = ['false', 'no', 'n', '0'];

/** One CSV label -> the training label for that question type, or an error message. */
export function csvLabel(q: QuestionDto, raw: string): { label?: boolean | string | number; error?: string } {
  const v = raw.trim();
  if (q.type === 'noul') {
    if (TRUE.includes(v.toLowerCase())) return { label: true };
    if (FALSE.includes(v.toLowerCase())) return { label: false };
    return { error: `"${v}" is not yes/no` };
  }
  const options = q.options ?? [];
  if (q.type === 'choice') return options.includes(v) ? { label: v } : { error: `"${v}" is not one of the options` };
  const level = /^\d+$/.test(v) ? Number(v) : options.indexOf(v);
  return level >= 0 && level < options.length ? { label: level } : { error: `"${v}" is not a level (0-${options.length - 1} or its text)` };
}

/** An upload -> labelled training JSONL lines (model-service data.py format). Throws a readable error on bad data. */
export function toTrainingLines(dto: Pick<CreateSpecialistDto, 'format' | 'data' | 'question'>): string[] {
  const errors: string[] = [];
  const lines: string[] = [];
  if (dto.format === 'jsonl') {
    dto.data.split(/\r?\n/).forEach((line, i) => {
      if (!line.trim()) return;
      try {
        const r = JSON.parse(line);
        const qs = r?.questions && typeof r.questions === 'object' ? Object.values(r.questions) : [];
        if (r?.state === undefined || !qs.length) throw new Error('needs a state and a non-empty questions object');
        if (qs.some((q) => (q as { label?: unknown }).label === undefined)) throw new Error('every question needs a label');
        lines.push(JSON.stringify(r));
      } catch (e) {
        errors.push(`line ${i + 1}: ${(e as Error).message}`);
      }
    });
  } else {
    const q = dto.question;
    if (!q) throw new BadRequestException('a CSV upload needs the question it answers');
    if (q.type !== 'noul' && (q.options?.length ?? 0) < 2) throw new BadRequestException('choice and score questions need at least 2 options');
    const rows = parseCsv(dto.data);
    const header = rows[0]?.map((h) => h.trim().toLowerCase()) ?? [];
    const [ti, li] = [header.indexOf('text'), header.indexOf('label')];
    if (ti < 0 || li < 0) throw new BadRequestException('the CSV needs a header row with the columns text,label');
    const criteria =
      q.type === 'noul' ? null : q.type === 'choice' ? Object.fromEntries(q.options!.map((o) => [o, null])) : q.options;
    rows.slice(1).forEach((row, i) => {
      const { label, error } = csvLabel(q, row[li] ?? '');
      if (error || !(row[ti] ?? '').trim()) {
        errors.push(`row ${i + 2}: ${error ?? 'empty text'}`);
        return;
      }
      lines.push(JSON.stringify({ state: row[ti], questions: { q: { type: q.type, instructions: q.instructions, criteria, label } } }));
    });
  }
  if (errors.length) throw new BadRequestException(`${errors.length} bad rows: ${errors.slice(0, 5).join('; ')}`);
  if (lines.length < MIN_EXAMPLES) throw new BadRequestException(`need at least ${MIN_EXAMPLES} labelled examples, got ${lines.length}`);
  return lines;
}

@Controller('specialists')
export class SpecialistsController {
  constructor(private readonly redis: RedisService) {}

  private async jobs() {
    const ids = await this.redis.client.lrange(JOB_LIST, 0, 19);
    const jobs = await Promise.all(
      ids.map(async (id) => {
        const raw = await this.redis.client.hgetall(jobKey(id));
        return { id, ...Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, JSON.parse(v)])) };
      }),
    );
    return jobs.filter((j) => Object.keys(j).length > 1);
  }

  @Get()
  async list() {
    const served = await callModel('/v1/specialists').catch(() => ({ specialists: [] }));
    return { specialists: served.specialists ?? [], jobs: await this.jobs() };
  }

  /** Upload labelled examples and train a specialist on them (the trainer worker picks the job up). */
  @Throttle({ default: { limit: 10, ttl: 60_000 } })
  @Post()
  async create(@Body() dto: CreateSpecialistDto) {
    if (RESERVED.includes(dto.name)) throw new BadRequestException(`"${dto.name}" is reserved`);
    const lines = toTrainingLines(dto);
    const id = randomUUID();
    await mkdir(config.uploadsDir, { recursive: true });
    const data = join(config.uploadsDir, `${id}.jsonl`);
    await writeFile(data, lines.join('\n') + '\n', 'utf8');
    const job = { id, op: 'train', name: dto.name, description: dto.description, tier: dto.tier, data, epochs: dto.epochs ?? 3 };
    const state = { status: 'queued', name: dto.name, tier: dto.tier, examples: lines.length, created: Date.now() / 1000 };
    await this.redis.client
      .multi()
      .hset(jobKey(id), Object.fromEntries(Object.entries(state).map(([k, v]) => [k, JSON.stringify(v)])))
      .lpush(JOB_LIST, id)
      .ltrim(JOB_LIST, 0, 49)
      .lpush(JOBS, JSON.stringify(job))
      .exec();
    return { id, examples: lines.length, held_out: Math.max(1, Math.round(lines.length * 0.2)) };
  }

  @Roles('admin')
  @Delete(':name')
  async remove(@Param('name') name: string) {
    if (!/^[a-z0-9][a-z0-9-]{1,39}$/.test(name)) throw new BadRequestException('invalid name');
    const id = randomUUID();
    await this.redis.client.lpush(JOBS, JSON.stringify({ id, op: 'delete', name }));
    return { ok: true, id };
  }
}
