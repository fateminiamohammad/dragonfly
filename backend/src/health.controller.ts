import { Controller, Get } from '@nestjs/common';
import { Public } from './auth/auth.guard';

@Controller('health')
export class HealthController {
  @Public()
  @Get()
  health() {
    return { status: 'ok', timestamp: new Date().toISOString() };
  }
}
