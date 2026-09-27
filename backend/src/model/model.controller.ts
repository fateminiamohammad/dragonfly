import { BadGatewayException, Body, Controller, Get, HttpException, Post } from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import { config } from '../config';

async function callModel(path: string, init?: RequestInit) {
  const headers: Record<string, string> = { 'content-type': 'application/json' };
  if (config.modelService.apiKey) headers.authorization = `Bearer ${config.modelService.apiKey}`;
  let res: Response;
  try {
    res = await fetch(`${config.modelService.url}${path}`, { ...init, headers, signal: AbortSignal.timeout(30_000) });
  } catch (e) {
    throw new BadGatewayException(`model-service unreachable: ${(e as Error).message}`);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new HttpException(body, res.status);
  return body;
}

/** Model info and a logged-in playground for the UI. Clients call the model-service directly at /v1, not this. */
@Controller('model')
export class ModelController {
  @Get()
  info() {
    return callModel('/v1/models');
  }

  @Throttle({ default: { limit: 120, ttl: 60_000 } })
  @Post('playground')
  playground(@Body() request: Record<string, unknown>) {
    return callModel('/v1/systemone', { method: 'POST', body: JSON.stringify(request) });
  }
}
