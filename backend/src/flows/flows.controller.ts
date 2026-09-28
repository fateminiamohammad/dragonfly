import { BadRequestException, Body, Controller, Delete, Get, NotFoundException, Param, Post, Put } from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import { Allow, IsObject } from 'class-validator';
import { callModel } from '../model/model.controller';
import { RedisService } from '../redis/redis.service';

/** Saved flows: the model-service reads this hash (dragonfly/flows.py, RedisFlowStore). */
export const FLOWS = 'dragonfly:flows';
const NAME = /^[a-z0-9][a-z0-9-]{0,63}$/;

interface Step {
  model?: string;
  state?: unknown;
  questions?: Record<string, unknown>;
  next?: { if?: string; to?: string }[];
}

/** Structural checks with readable errors; the model-service re-validates conditions when it runs the flow. */
export function checkFlow(flow: { start?: string; steps?: Record<string, Step> }): string[] {
  const errors: string[] = [];
  const steps = flow.steps;
  if (!steps || typeof steps !== 'object' || !Object.keys(steps).length) return ['a flow needs a non-empty "steps" object'];
  if (flow.start && !(flow.start in steps)) errors.push(`start step "${flow.start}" is not defined`);
  for (const [name, step] of Object.entries(steps)) {
    if (!step?.questions || typeof step.questions !== 'object' || !Object.keys(step.questions).length) {
      errors.push(`step "${name}" needs questions`);
    }
    for (const edge of step?.next ?? []) {
      if (!edge.to || !(edge.to in steps)) errors.push(`step "${name}" goes to undefined step "${edge.to}"`);
    }
  }
  return errors;
}

class SaveFlowDto {
  @IsObject()
  flow: { start?: string; steps?: Record<string, Step> };
}

class RunFlowDto {
  @Allow()
  state: unknown;
}

class TryFlowDto {
  @IsObject()
  flow: Record<string, unknown>;

  @Allow()
  state: unknown;
}

@Controller('flows')
export class FlowsController {
  constructor(private readonly redis: RedisService) {}

  @Get()
  async list() {
    const all = await this.redis.client.hgetall(FLOWS);
    return { flows: Object.entries(all).map(([name, raw]) => ({ name, flow: JSON.parse(raw) })) };
  }

  @Put(':name')
  async save(@Param('name') name: string, @Body() dto: SaveFlowDto) {
    if (!NAME.test(name)) throw new BadRequestException('name: lowercase letters, digits and dashes');
    const errors = checkFlow(dto.flow);
    if (errors.length) throw new BadRequestException(errors.join('; '));
    await this.redis.client.hset(FLOWS, name, JSON.stringify({ ...dto.flow, name }));
    return { ok: true, name };
  }

  @Delete(':name')
  async remove(@Param('name') name: string) {
    if (!(await this.redis.client.hdel(FLOWS, name))) throw new NotFoundException(`no flow named ${name}`);
    return { ok: true };
  }

  @Throttle({ default: { limit: 60, ttl: 60_000 } })
  @Post(':name/run')
  run(@Param('name') name: string, @Body() dto: RunFlowDto) {
    return callModel(`/v1/flows/${encodeURIComponent(name)}/run`, {
      method: 'POST',
      body: JSON.stringify({ state: dto.state, mermaid: true }),
    });
  }

  /** Run an unsaved flow from the editor. */
  @Throttle({ default: { limit: 60, ttl: 60_000 } })
  @Post('try')
  try(@Body() dto: TryFlowDto) {
    return callModel('/v1/flows/run', { method: 'POST', body: JSON.stringify({ flow: dto.flow, state: dto.state, mermaid: true }) });
  }
}
