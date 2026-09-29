import { BadRequestException, Body, Controller, Get, Param, Post, Query } from '@nestjs/common';
import { Throttle } from '@nestjs/throttler';
import { IsArray, ArrayMaxSize, ArrayMinSize } from 'class-validator';
import { callModel } from '../model/model.controller';

class BatchDto {
  @IsArray()
  @ArrayMinSize(1)
  @ArrayMaxSize(10000)
  requests: Record<string, unknown>[];
}

/** Batch decisions from the UI: proxies the model-service's /v1/batch/jobs (runs at batch priority). */
@Controller('batch')
export class BatchController {
  @Throttle({ default: { limit: 10, ttl: 60_000 } })
  @Post('jobs')
  start(@Body() dto: BatchDto) {
    return callModel('/v1/batch/jobs', { method: 'POST', body: JSON.stringify({ requests: dto.requests }) });
  }

  @Get('jobs/:id')
  status(@Param('id') id: string, @Query('results') results?: string) {
    if (!/^[a-f0-9]{32}$/.test(id)) throw new BadRequestException('invalid job id');
    return callModel(`/v1/batch/jobs/${id}?results=${results === 'false' ? 'false' : 'true'}`);
  }
}
