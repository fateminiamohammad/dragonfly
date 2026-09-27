/** Environment configuration, read once. Names follow docker/.env.example. */
function required(name: string, fallback?: string): string {
  const value = process.env[name] ?? fallback;
  if (value === undefined || value === '') throw new Error(`environment variable ${name} is required`);
  return value;
}

export const config = {
  port: Number(process.env.BACKEND_PORT ?? 3000),
  db: {
    host: process.env.DB_HOST ?? 'localhost',
    port: Number(process.env.DB_PORT ?? 5432),
    username: process.env.POSTGRES_USER ?? 'dragonfly',
    password: process.env.POSTGRES_PASSWORD ?? 'dragonfly',
    database: process.env.POSTGRES_DB ?? 'dragonfly',
  },
  redisUrl: process.env.REDIS_URL ?? 'redis://localhost:6379',
  jwt: {
    secret: () => required('JWT_SECRET'),
    expiresIn: process.env.JWT_EXPIRES_IN ?? '12h',
  },
  admin: {
    email: process.env.ADMIN_EMAIL,
    password: process.env.ADMIN_PASSWORD,
  },
  modelService: {
    url: process.env.MODEL_SERVICE_URL ?? 'http://localhost:8000',
    // a key the model-service accepts (listed in DRAGONFLY_API_KEYS), for the playground and model info
    apiKey: process.env.MODEL_SERVICE_API_KEY ?? '',
  },
  corsOrigins: (process.env.CORS_ORIGINS ?? '').split(',').filter(Boolean),
};
