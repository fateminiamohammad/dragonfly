import { MigrationInterface, QueryRunner } from 'typeorm';

export class Init1790000000000 implements MigrationInterface {
  name = 'Init1790000000000';

  public async up(q: QueryRunner): Promise<void> {
    await q.query(`CREATE EXTENSION IF NOT EXISTS "pgcrypto"`);
    await q.query(`
      CREATE TABLE "users" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        "email" varchar NOT NULL UNIQUE,
        "password_hash" varchar NOT NULL,
        "role" varchar NOT NULL DEFAULT 'member',
        "created_at" timestamptz NOT NULL DEFAULT now()
      )`);
    await q.query(`
      CREATE TABLE "api_keys" (
        "id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        "name" varchar NOT NULL,
        "prefix" varchar NOT NULL,
        "digest" varchar NOT NULL,
        "user_id" uuid NOT NULL REFERENCES "users"("id") ON DELETE CASCADE,
        "created_at" timestamptz NOT NULL DEFAULT now(),
        "revoked_at" timestamptz
      )`);
    await q.query(`CREATE UNIQUE INDEX "idx_api_keys_digest" ON "api_keys" ("digest")`);
    await q.query(`CREATE INDEX "idx_api_keys_user" ON "api_keys" ("user_id")`);
  }

  public async down(q: QueryRunner): Promise<void> {
    await q.query(`DROP TABLE "api_keys"`);
    await q.query(`DROP TABLE "users"`);
  }
}
