import 'reflect-metadata';
import { DataSource, DataSourceOptions } from 'typeorm';
import { config } from './config';
import { ApiKey } from './entities/api-key.entity';
import { User } from './entities/user.entity';
import { Init1790000000000 } from './migrations/1790000000000-Init';

/** Schema changes only through migrations (never synchronize), run automatically at startup. */
export const dataSourceOptions: DataSourceOptions = {
  type: 'postgres',
  ...config.db,
  entities: [User, ApiKey],
  migrations: [Init1790000000000],
  synchronize: false,
  migrationsRun: true,
};

export default new DataSource(dataSourceOptions);
