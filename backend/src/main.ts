import { Logger, ValidationPipe } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { NestExpressApplication } from '@nestjs/platform-express';
import { AppModule } from './app.module';
import { config } from './config';

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);
  // images/audio arrive base64-encoded in JSON (playground, media page): the 100 kB default is far too small
  app.useBodyParser('json', { limit: '30mb' });
  app.setGlobalPrefix('api');
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, forbidNonWhitelisted: true, transform: true }));
  if (config.corsOrigins.length) app.enableCors({ origin: config.corsOrigins });
  app.enableShutdownHooks();
  await app.listen(config.port, '0.0.0.0');
  new Logger('bootstrap').log(`backend listening on :${config.port}`);
}

void bootstrap();
