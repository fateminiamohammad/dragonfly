import { Module } from '@nestjs/common';
import { APP_GUARD } from '@nestjs/core';
import { JwtModule } from '@nestjs/jwt';
import { ThrottlerGuard, ThrottlerModule } from '@nestjs/throttler';
import { TypeOrmModule } from '@nestjs/typeorm';
import { AuthController } from './auth/auth.controller';
import { JwtAuthGuard } from './auth/auth.guard';
import { AuthService } from './auth/auth.service';
import { config } from './config';
import { dataSourceOptions } from './data-source';
import { ApiKey } from './entities/api-key.entity';
import { User } from './entities/user.entity';
import { HealthController } from './health.controller';
import { KeysController } from './keys/keys.controller';
import { KeysService } from './keys/keys.service';
import { ModelController } from './model/model.controller';
import { RedisService } from './redis/redis.service';
import { ReviewController } from './review/review.controller';
import { UsageController } from './usage/usage.controller';

@Module({
  imports: [
    TypeOrmModule.forRoot(dataSourceOptions),
    TypeOrmModule.forFeature([User, ApiKey]),
    JwtModule.registerAsync({
      global: true,
      useFactory: () => ({ secret: config.jwt.secret(), signOptions: { expiresIn: config.jwt.expiresIn } }),
    }),
    ThrottlerModule.forRoot([{ ttl: 60_000, limit: 300 }]),
  ],
  controllers: [HealthController, AuthController, KeysController, UsageController, ModelController, ReviewController],
  providers: [
    AuthService,
    KeysService,
    RedisService,
    { provide: APP_GUARD, useClass: ThrottlerGuard },
    { provide: APP_GUARD, useClass: JwtAuthGuard },
  ],
})
export class AppModule {}
