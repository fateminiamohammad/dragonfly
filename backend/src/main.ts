import { Logger, ValidationPipe } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { config } from './config';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.setGlobalPrefix('api');
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true, transform: true }));
  if (config.corsOrigins.length) app.enableCors({ origin: config.corsOrigins });
  app.enableShutdownHooks();
  await app.listen(config.port, '0.0.0.0');
  new Logger('bootstrap').log(`backend listening on :${config.port}`);
}

void bootstrap();
